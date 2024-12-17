import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Transformer, ModelArgs
from tokenizer import  Tokenizer
from utils import save_checkpoint
from tokenizer import Tokenizer
import os
import sentencepiece as spm
from typing import Optional,List
from utils import load_model


def preprocess_dataset(dataset: Dataset, batch_size: int,model_args: ModelArgs,tokenizer_model: str)->DataLoader:

    tokenizer = Tokenizer(tokenizer_model)


    def collate_fn(batch:List[dict]):
        # Combine all input_ids into one dictionary
        combined = {
            'input_ids': [item['input_ids'][:model_args.max_seq_len] for item in batch]  # Truncate to max_seq_len
        }
        # Use the tokenizer's pad method to pad the sequences
        padded = tokenizer.pad(combined)
            
        # Create targets by shifting input_ids right by 1
        input_ids = padded['input_ids']
        targets = input_ids.clone()
        targets = torch.roll(targets, -1, dims=1)
        targets[:, -1] = tokenizer.pad_id
            
        return {
            'input_ids': input_ids,
            'targets': targets
        }


    def tokenize_function(examples: dict):
        tokenized = [tokenizer.encode(text, bos=True, eos=True) for text in examples['text']]
        return {'input_ids': tokenized}
    
    
    tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
        )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_dataloader



def train(num_steps: int, learning_rate: float, model: Transformer, train_dataloader: DataLoader):
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(num_steps), desc="Training")
    dataloader_iterator = iter(train_dataloader)

    for step in progress_bar:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_dataloader)
            batch = next(dataloader_iterator)
        
        # Move batch to device and extract input_ids and targets
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass with targets
        outputs = model(input_ids, targets=targets)
        loss = model.last_loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f"Step {step + 1}, Average loss: {avg_loss}")
            total_loss = 0

        if (step + 1) % 1000 == 0: 
            save_checkpoint(
                model, 
                optimizer,
                step,
                loss.item(),
                f'checkpoint_step_{step}.pth'
            )

    return model

def generate_text(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text from a prompt."""
    # Encode the prompt
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(encoded).unsqueeze(0).to(device)  # Create a batch of size 1
    
    # Move model to device
    model = model.to(device)
    
    # Generate tokens
    with torch.no_grad():
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    return generated_text

def train_workflow(num_steps: int, batch_size: int, learning_rate: float,vocab_size: int, dataset_name: str, dim: int, n_layers: int, n_heads: int, max_seq_len: int, tokenizer_model: str, output_path: str, dataset_size: Optional[int], dataset_data_dir: Optional[str]):
    ds = load_dataset(dataset_name, data_dir=dataset_data_dir)
    ds['train'] = ds['train'].select(range(dataset_size))
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,  # Make sure this matches tokenizer
        max_seq_len=max_seq_len,
    )
    model = Transformer(model_args)


    train_dataloader = preprocess_dataset(ds['train'], batch_size=batch_size, model_args=model_args,tokenizer_model=tokenizer_model)


    model = train(num_steps=num_steps, learning_rate=learning_rate, model=model, train_dataloader=train_dataloader)

    torch.save(model.state_dict(), output_path)

def train_vocab(vocab_size: int, dataset_name: str, dataset_data_dir: Optional[str], dataset_size: Optional[int], model_prefix: str = "tokenizer"):
    # Convert dataset to list of strings
    ds = load_dataset(dataset_name, data_dir=dataset_data_dir)
    ds['train'] = ds['train'].select(range(dataset_size))
    train_text = ds["train"]['text']
    
    # Create a temporary file to write the text data
    with open("temp_train_data.txt", "w", encoding="utf-8") as f:
        for text in train_text:
            f.write(text + "\n")
    
    # Train the tokenizer using the temporary file
    spm.SentencePieceTrainer.train(
        input="temp_train_data.txt",
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_piece="[PAD]",
    )
    
    os.remove("temp_train_data.txt")

def inference_workflow(model_path: str, prompt: str, dim: int, n_layers: int, n_heads: int, vocab_size: int, max_seq_len: int):
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model = load_model(model_path, model_args)
    tokenizer = Tokenizer()
    return generate_text(model=model, tokenizer=tokenizer, prompt=prompt)



train_vocab(
    vocab_size=4096, 
    dataset_name="ai4bharat/sangraha", 
    dataset_data_dir="verified/ben", 
    dataset_size=1000,
    model_prefix="bangla_tokenizer"
)
train_workflow(num_steps=100, batch_size=64, learning_rate=0.001, vocab_size=4096, dataset_name="ai4bharat/sangraha", dim=64, n_layers=5, n_heads=8, max_seq_len=1024, tokenizer_model="bangla_tokenizer.model", output_path="bangla_model.pth", dataset_size=1000, dataset_data_dir="verified/ben")

print(inference_workflow(
    model_path="bangla_model.pth",
    prompt="আপনি কেমন আছেন?",
    dim=64,
    n_layers=5,
    n_heads=8,
    vocab_size=4096,
    max_seq_len=1024
))
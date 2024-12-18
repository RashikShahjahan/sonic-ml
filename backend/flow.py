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
from typing import List
from utils import load_model
from flytekit import task, workflow


@task
def preprocess_dataset(dataset: Dataset, batch_size: int, max_seq_len: int,tokenizer_model: str)->DataLoader:
    tokenizer = Tokenizer(tokenizer_model)

    def collate_fn(batch:List[dict]):
        # Combine all input_ids into one dictionary
        combined = {
            'input_ids': [item['input_ids'][:max_seq_len] for item in batch]  # Truncate to max_seq_len
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
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_dataloader



@task
def train(num_steps: int, learning_rate: float, dim: int, n_layers: int, n_heads: int, vocab_size: int, max_seq_len: int, train_dataloader: DataLoader,model_id: str, gradient_accumulation_steps: int):
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model = Transformer(model_args)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(num_steps), desc="Training")
    dataloader_iterator = iter(train_dataloader)
    
    optimizer.zero_grad()
    
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
        model(input_ids, targets=targets)
        # Scale the loss by gradient_accumulation_steps
        loss = model.last_loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
        
        # Backward pass
        loss.backward()
        
        # Only step and zero grad after accumulating enough gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})  # Scale back for display
        
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f"Step {step + 1}, Average loss: {avg_loss}")
            total_loss = 0

        if (step + 1) % 100 == 0: 
            save_checkpoint(
                model, 
                optimizer,
                step,
                loss.item(),
                f'checkpoints/{model_id}_checkpoint_step_{step}.pth',
            )

    return model

@task
def generate_text(
    model_path: str,
    tokenizer_model: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text from a prompt."""
    # Encode the prompt
    tokenizer = Tokenizer(tokenizer_model)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(encoded).unsqueeze(0).to(device)  # Create a batch of size 1
    
    # Move model to device
    model = load_model(model_path).to(device)
    
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

@task
def load_and_prepare_dataset(dataset_name: str, dataset_size: int, dataset_data_dir:str) -> Dataset:
    ds = load_dataset(dataset_name, data_dir=dataset_data_dir)
    if dataset_size:
        ds['train'] = ds['train'].select(range(dataset_size))
    return ds['train']

@workflow
def train_workflow(num_steps: int, batch_size: int, learning_rate: float, vocab_size: int, dataset_name: str, 
                  dim: int, n_layers: int, n_heads: int, max_seq_len: int, tokenizer_model: str, 
                  model_id: str, dataset_size: int, dataset_data_dir: str, gradient_accumulation_steps: int):
    # First load the dataset as a separate task
    dataset = load_and_prepare_dataset(
        dataset_name=dataset_name, 
        dataset_size=dataset_size, 
        dataset_data_dir=dataset_data_dir
    )
    

    train_dataloader = preprocess_dataset(
        dataset=dataset, 
        batch_size=batch_size, 
        max_seq_len=max_seq_len,
        tokenizer_model=tokenizer_model
    )

    train(
        num_steps=num_steps, 
        learning_rate=learning_rate, 
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        train_dataloader=train_dataloader,
        model_id=model_id,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

@workflow
def train_vocab(vocab_size: int, dataset_name: str, dataset_data_dir: str, 
                dataset_size: int, model_prefix: str = "tokenizer"):
    # Load dataset as a separate task
    dataset = load_and_prepare_dataset(
        dataset_name=dataset_name, 
        dataset_size=dataset_size, 
        dataset_data_dir=dataset_data_dir
    )
    
    # Create a new task for tokenizer training
    return train_tokenizer(dataset=dataset, vocab_size=vocab_size, model_prefix=model_prefix)

@task
def train_tokenizer(dataset: Dataset, vocab_size: int, model_prefix: str = "tokenizer") -> None:
    # Create a temporary file to write the text data
    with open("temp_train_data.txt", "w", encoding="utf-8") as f:
        for text in dataset['text']:
            f.write(text + "\n")
    
    # Train the tokenizer using the temporary file
    spm.SentencePieceTrainer.train(
        input="temp_train_data.txt",
        model_prefix=f"tokenizers/{model_prefix}",
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

@workflow
def inference_workflow(model_path: str, tokenizer_model: str, prompt: str):
     print(generate_text(model_path=model_path, tokenizer_model=tokenizer_model, prompt=prompt))


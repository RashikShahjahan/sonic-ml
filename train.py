import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import  get_linear_schedule_with_warmup
from tqdm import tqdm
from model import Transformer, ModelArgs
from tokenizer import  Tokenizer, train_vocab

# Load dataset
ds = load_dataset("ai4bharat/sangraha", data_dir="verified/ben")
ds['train'] = ds['train'].select(range(10000))


# First, train the vocabulary on the dataset
vocab_size = 4096  # Keep same as model
# Train vocab on the dataset
train_vocab(vocab_size, ds)
tokenizer = Tokenizer()


# Initialize model with the same vocab size and padding token ID
model_args = ModelArgs(
    dim=64,
    n_layers=5,
    n_heads=8,
    vocab_size=vocab_size,  # Make sure this matches tokenizer
    max_seq_len=1024,
)
model = Transformer(model_args)

# Tokenize the dataset
def tokenize_function(examples: dict):
    tokenized = [tokenizer.encode(text, bos=True, eos=True) for text in examples['text']]
    # Add warning for long sequences
    for tokens in tokenized:
        if len(tokens) > model_args.max_seq_len:
            print(f"Warning: Found sequence of length {len(tokens)} > max_seq_len {model_args.max_seq_len}")
    return {'input_ids': tokenized}


def preprocess_dataset(dataset: Dataset, batch_size=16)->DataLoader:
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )

    def collate_fn(batch):
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

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_dataloader

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train(num_steps=1000, batch_size=64, learning_rate=1e-3, warmup_steps=1000):

    train_dataloader = preprocess_dataset(ds['train'], batch_size=batch_size)

    # Add detailed debugging for the first batch
    first_batch = next(iter(train_dataloader))
    input_ids = first_batch['input_ids']
    print(f"Input shape: {input_ids.shape}")
    print(f"Max token ID: {input_ids.max().item()}")
    print(f"Min token ID: {input_ids.min().item()}")
    print(f"Unique token IDs: {torch.unique(input_ids).tolist()}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )

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
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Optional: Print average loss every N steps
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f"Step {step + 1}, Average loss: {avg_loss}")
            total_loss = 0

        if (step + 1) % 1000 == 0:  # Save every 1000 steps
            save_checkpoint(
                model, 
                optimizer,
                step,
                loss.item(),
                f'checkpoint_step_{step}.pth'
            )

    # Save the model weights at the end of training
    torch.save(model.state_dict(), 'transformer_weights.pth')
    return model

# Let's also check the token IDs in the first batch
train_dataloader = preprocess_dataset(ds['train'], batch_size=64)
first_batch = next(iter(train_dataloader))
print(f"Max token ID in first batch: {first_batch['input_ids'].max()}")
print(f"Min token ID in first batch: {first_batch['input_ids'].min()}")

model = train(num_steps=1000)


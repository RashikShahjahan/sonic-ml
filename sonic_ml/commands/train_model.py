import torch
import os
from tqdm import tqdm
from typing import List
import datasets
from sonic_ml.architectures.llama2 import Transformer, ModelArgs
from sonic_ml.utils.utils import save_checkpoint, preprocess_dataset
from datasets import Dataset
from sonic_ml.utils.utils import load_and_prepare_dataset
from flytekit import task, workflow

@task   
def train(num_steps: int, learning_rate: float, dim: int, n_layers: int, n_heads: int, 
          vocab_size: int, max_seq_len: int, model_id: str, 
          gradient_accumulation_steps: int, dataset_path: str, chunk_size: int,
          batch_size: int, tokenizer_prefix: str, resume_from_checkpoint: bool = False):
    """Train a Transformer model on chunked dataset with gradient accumulation.
    
    Args:
        num_steps (int): Total number of training steps
        learning_rate (float): Learning rate for the optimizer
        dim (int): Model dimension/embedding size
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        vocab_size (int): Size of the vocabulary
        max_seq_len (int): Maximum sequence length for input texts
        model_id (str): Unique identifier for the model (used in checkpoint names)
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        dataset_path (str): Path to the dataset on disk
        chunk_size (int): Number of samples per chunk
        batch_size (int): Number of samples per batch
        tokenizer_prefix (str): Prefix for the tokenizer model file
        resume_from_checkpoint (bool, optional): Whether to resume from a previous checkpoint. Defaults to False.
    
    Returns:
        Transformer: The trained model
        
    The function implements the following training loop:
    1. Initializes model, optimizer, and loads first data chunk
    2. For each step:
        - Fetches next batch or loads new chunk if needed
        - Performs forward and backward passes
        - Updates model weights every gradient_accumulation_steps
        - Saves checkpoints every 100 steps
    3. Uses gradient accumulation for effective larger batch sizes
    4. Processes dataset in chunks to handle large datasets
    """
    # Move dataset loading into the task
    current_chunk, chunk_indices = load_and_prepare_dataset(dataset_path=dataset_path, chunk_size=chunk_size)
    
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    
    model_args = ModelArgs(
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        vocab_size=vocab_size, max_seq_len=max_seq_len,
    )
    model = Transformer(model_args)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate, 
        betas=(0.9, 0.95),
        device_type='cuda' if torch.cuda.is_available() 
                   else 'mps' if torch.backends.mps.is_available() 
                   else 'cpu'
    )
    
    # Load checkpoint if resuming
    start_step = 0
    if resume_from_checkpoint:
        # Find the latest checkpoint
        checkpoint_dir = f'checkpoints/{model_id}'
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'checkpoint_step_')]
        if checkpoints:
            # Get the checkpoint with the highest step number
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_step_')[1].split('.')[0]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path,weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            print(f"Resuming from step {start_step}")
            # When resuming, num_steps represents additional steps
            num_steps = start_step + num_steps

    model.to(device)
    
    
    # Initial preprocessing of first chunk
    train_dataloader = preprocess_dataset(
        dataset=current_chunk,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tokenizer_prefix=tokenizer_prefix
    )
    
    dataloader_iterator = iter(train_dataloader)
    current_chunk_idx = 1

    # Training loop
    model.train()
    total_loss = 0
    print(f"{start_step} {num_steps} ")
    progress_bar = tqdm(range(start_step, num_steps), desc="Training")
    
    optimizer.zero_grad()
    
    for step in progress_bar:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            if current_chunk_idx >= len(chunk_indices):
                print("\nReached end of dataset, restarting from first chunk")
                current_chunk_idx = 0  # Reset to beginning
                
            start_idx = chunk_indices[current_chunk_idx]
            end_idx = chunk_indices[current_chunk_idx + 1] if current_chunk_idx + 1 < len(chunk_indices) else None
            
            full_dataset = datasets.load_from_disk(dataset_path)
            if isinstance(full_dataset, datasets.DatasetDict):
                split_name = 'train' if 'train' in full_dataset else list(full_dataset.keys())[0]
                full_dataset = full_dataset[split_name]

            current_chunk = full_dataset.select(range(start_idx, end_idx) if end_idx else range(start_idx, len(full_dataset)))
            train_dataloader = preprocess_dataset(
                dataset=current_chunk, 
                batch_size=batch_size, 
                max_seq_len=max_seq_len, 
                tokenizer_prefix=tokenizer_prefix
            )
            dataloader_iterator = iter(train_dataloader)
            current_chunk_idx += 1
            print(f"\nLoading chunk {current_chunk_idx} of {len(chunk_indices)}")
            
            batch = next(dataloader_iterator)
        
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        model(input_ids, targets=targets)
        loss = model.last_loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps
        
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        if (step + 1) % 500 == 0:
            avg_loss = total_loss / 500
            print(f"Step {step + 1}, Average loss: {avg_loss}")
            os.makedirs(f'checkpoints/{model_id}', exist_ok=True)
            total_loss = 0
            save_checkpoint(
                model, optimizer, step, loss.item(),
                f'checkpoints/{model_id}/checkpoint_step_{step}.pth',
            )

    return model

@workflow
def train_workflow(num_steps: int, batch_size: int, learning_rate: float, vocab_size: int,
                  dim: int, n_layers: int, n_heads: int, max_seq_len: int, tokenizer_prefix: str, 
                  model_id: str, dataset_path: str, gradient_accumulation_steps: int, chunk_size: int,
                  resume_from_checkpoint: bool = False):
    """Orchestrates the complete training workflow for the transformer model.
    
    Args:
        num_steps (int): Total number of training steps
        batch_size (int): Number of samples per batch
        learning_rate (float): Learning rate for the optimizer
        vocab_size (int): Size of the vocabulary
        dim (int): Model dimension/embedding size
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        max_seq_len (int): Maximum sequence length for input texts
        tokenizer_prefix (str): Prefix for the tokenizer model file
        model_id (str): Unique identifier for the model (used in checkpoint names)
        dataset_path (str): Path to the dataset on disk
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        chunk_size (int): Number of samples per chunk
    """
    # Remove dataset loading from workflow
    train(
        num_steps=num_steps, 
        learning_rate=learning_rate, 
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_id=model_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_path=dataset_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        tokenizer_prefix=tokenizer_prefix,
        resume_from_checkpoint=resume_from_checkpoint
    )
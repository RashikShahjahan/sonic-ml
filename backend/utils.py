import torch
from model import Transformer
from model import ModelArgs
from torch.utils.data import DataLoader
from datasets import Dataset
from tokenizer import Tokenizer
import os
from typing import List
import datasets

# -----------------------------------------------------------------------------
# sampling utils

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def save_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer, step: int, loss: float, path: str):
    """Save model checkpoint including model arguments for complete restoration"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_args': model.params.__dict__,  # Save model arguments as dictionary
    }, path)



def load_model(model_path: str) -> Transformer:
    """
    Load a trained transformer model from a given path.
    
    Args:
        model_path: Path to the saved model state dict
        
    Returns:
        Loaded Transformer model
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path,weights_only=True)
    print(checkpoint.keys())
    
    # Initialize model with saved args
    model_args = ModelArgs(**checkpoint['model_args'])
    model = Transformer(model_args)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def preprocess_dataset(dataset: Dataset, batch_size: int, max_seq_len: int, tokenizer_prefix: str) -> DataLoader:
    """Preprocess a raw dataset into a DataLoader for training.
    
    Args:
        dataset (Dataset): The raw dataset containing text samples
        batch_size (int): Number of samples per batch
        max_seq_len (int): Maximum sequence length for truncation
        tokenizer_prefix (str): Prefix for the tokenizer model file
        
    Returns:
        DataLoader: A DataLoader instance that yields batches of:
            - input_ids: Tensor of tokenized and padded input sequences
            - targets: Tensor of shifted input sequences for next-token prediction
            
    The function performs the following steps:
    1. Tokenizes the raw text using the specified tokenizer
    2. Pads sequences to the same length within each batch
    3. Creates target sequences by shifting input sequences
    4. Returns a DataLoader with multi-processing support
    """
    # Construct full path to tokenizer model
    tokenizer_path = os.path.join("tokenizers", f"{tokenizer_prefix}.model")

    tokenizer = Tokenizer(tokenizer_path)

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




import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from sonic_ml.tokenizer.tokenizer import Tokenizer
import os
from typing import List
import datasets
import torch.nn as nn

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

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, loss: float, path: str):
    """Save model checkpoint including model configuration for complete restoration"""

    config = model.config.to_dict() if hasattr(model.config, 'to_dict') else model.config
   

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'model_type': model.__class__.__name__,  # Save model type for reconstruction
    }, path)

def load_model(model_path: str) -> nn.Module:
    """
    Load a trained model from a given path.
    
    Args:
        model_path: Path to the saved model state dict
        
    Returns:
        Loaded model
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    # Get model type and config
    model_type = checkpoint['model_type']
    config = checkpoint['config']
    
    # Initialize appropriate model based on saved type
    if model_type == 'Llama':
        from sonic_ml.architectures.llama import Llama
        model = Llama(
            dim=config['hidden_size'],
            n_layers=config['num_hidden_layers'],
            n_heads=config['num_attention_heads'],
            vocab_size=config['vocab_size'],
            max_seq_len=config['max_position_embeddings']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
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
        num_workers=0 if os.name == 'posix' else os.cpu_count(),
        pin_memory=True,
        prefetch_factor=None if os.name == 'posix' else 2,
        persistent_workers=False if os.name == 'posix' else True
    )

    return train_dataloader

def load_and_prepare_dataset(dataset_path: str, chunk_size: int = 1000) -> tuple[Dataset, list[int]]:
    """Load a dataset from local storage and prepare it for chunked processing.
    
    Args:
        dataset_path (str): Path to the saved dataset on disk
        chunk_size (int, optional): Number of examples to include in each chunk. Defaults to 1000.
    
    Returns:
        tuple[Dataset, list[int]]: A tuple containing:
            - Dataset: The first chunk of the dataset
            - list[int]: List of starting indices for all chunks
            
    The function performs the following steps:
    1. Loads the complete dataset from disk
    2. Calculates chunk indices based on the dataset size and chunk_size
    3. Returns the first chunk and all chunk indices for subsequent loading
    
    Example:
        >>> first_chunk, indices = load_and_prepare_dataset(
        ...     dataset_path='data/processed/wikitext',
        ...     chunk_size=1000
        ... )
        >>> print(f"First chunk size: {len(first_chunk)}")
        >>> print(f"Total chunks: {len(indices)}")
    """
    # Load the dataset
    full_dataset = datasets.load_from_disk(dataset_path)
    
    # Get the 'train' split (or another appropriate split)
    if isinstance(full_dataset, datasets.DatasetDict):
        # Use 'train' split by default, or the first available split
        split_name = 'train' if 'train' in full_dataset else list(full_dataset.keys())[0]
        full_dataset = full_dataset[split_name]
    
    # Create chunks
    total_size = len(full_dataset)
    indices = list(range(0, total_size, chunk_size))
    
    # Return the first chunk initially
    return full_dataset.select(range(min(chunk_size, total_size))), indices
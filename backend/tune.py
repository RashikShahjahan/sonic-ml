import numpy as np

def calculate_chunk_size(max_available_memory_gb: float, 
                        batch_size: int, 
                        max_seq_len: int,
                       ) -> int:
    """Calculate optimal chunk size based on memory constraints.
    
    Args:
        max_available_memory_gb: Available GPU/RAM memory in GB
        batch_size: Training batch size
        max_seq_len: Maximum sequence length
        safety_factor: Fraction of memory to use (default 0.5 for headroom)
    
    Returns:
        int: Recommended chunk size
    """
    # Convert GB to bytes
    max_memory_bytes = max_available_memory_gb * 1024 * 1024 * 1024
    
    # Memory per sample (approximate)
    bytes_per_sample = max_seq_len * 4  # 4 bytes per token (float32)
    
    # Calculate chunk size leaving room for model, gradients etc
    chunk_size = int((max_memory_bytes) / (bytes_per_sample * batch_size))
    
    # Round to nearest 100 for clean numbers
    chunk_size = (chunk_size // 100) * 100
    
    return max(100, chunk_size)  # Minimum 100 samples per chunk


def calculate_num_parameters(vocab_size: int, dim: int, n_layers: int, n_heads: int, n_kv_heads: int = None, hidden_dim: int = None, multiple_of: int = 256) -> int:
    """Calculate the total number of parameters based on model architecture"""
    n_params = 0
    
    # Token embeddings
    n_params += vocab_size * dim  # tok_embeddings
    
    # Each transformer layer
    for _ in range(n_layers):
        # Attention
        n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        n_params += dim * (n_heads * dim // n_heads)  # wq
        n_params += dim * (n_kv_heads * dim // n_heads)    # wk
        n_params += dim * (n_kv_heads * dim // n_heads)    # wv
        n_params += (n_heads * dim // n_heads) * dim  # wo
        n_params += dim  # attention_norm
        
        # FeedForward
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        if hidden_dim is not None:
            hidden_dim = hidden_dim
            
        n_params += dim * hidden_dim  # w1
        n_params += hidden_dim * dim  # w2
        n_params += dim * hidden_dim  # w3
        n_params += dim  # ffn_norm
    
    # Final norm and output
    n_params += dim  # norm
    n_params += dim * vocab_size  # output (shared with tok_embeddings)
    
    return n_params


def calculate_model_memory(n_params: int, dtype_bytes: int = 4) -> float:
    """Calculate approximate GPU memory usage for model training in GB.
    
    Args:
        n_params: Number of model parameters
        dtype_bytes: Bytes per parameter (default 4 for float32)
        gradient_checkpointing: Whether gradient checkpointing is enabled
    
    Returns:
        float: Estimated GPU memory usage in GB
    """
    # Model weights
    model_memory = n_params * dtype_bytes
    
    # Optimizer states (Adam has 2 additional states per parameter)
    optimizer_memory = 2 * model_memory
    
    # Gradients
    gradient_memory = model_memory
    # Total memory in GB
    total_memory_gb = (model_memory + optimizer_memory + gradient_memory) / (1024 * 1024 * 1024)
    
    return total_memory_gb


def calculate_max_batch_size(max_available_memory_gb: float,
                           max_seq_len: int,
                           model_memory_gb: float,
                           safety_factor: float = 0.8) -> int:
    """Calculate maximum safe batch size based on available memory."""
    # Calculate remaining memory after model
    remaining_memory_gb = max(0, max_available_memory_gb - model_memory_gb)
    remaining_memory_bytes = remaining_memory_gb * safety_factor * 1024 * 1024 * 1024
    
    # Memory per sample (accounting for activations and temporary tensors)
    bytes_per_sample = max_seq_len * 4 * 3  # 4 bytes per token, multiplied by 3 for activations
    
    # Calculate max batch size
    max_batch_size = int(remaining_memory_bytes / bytes_per_sample)
    
    # Round down to nearest power of 2 for better performance
    if max_batch_size > 1:
        max_batch_size = 2 ** int(np.log2(max_batch_size))
    
    return max(1, max_batch_size)


def calculate_desired_batch_size(total_tokens: int, seq_len: int) -> int:
    """Calculate the desired batch size to achieve a specific total token count per update.
    
    Args:
        total_tokens: Desired total number of tokens per update
        seq_len: Sequence length
    
    Returns:
        int: Desired batch size
    """
    # Calculate the desired batch size
    desired_batch_size = total_tokens // (seq_len)

    # Round to nearest power of 2 for better performance
    if desired_batch_size > 1:
        desired_batch_size = 2 ** int(np.log2(desired_batch_size))
    
    return max(1, desired_batch_size)  # Ensure at least a batch size of 1


def calculate_grad_accum_steps(total_tokens: int, seq_len: int, batch_size: int) -> int:
    """Calculate the necessary gradient accumulation steps to achieve a specific total token count per update.
    
    Args:
        total_tokens: Desired total number of tokens per update
        seq_len: Sequence length
        batch_size: Batch size
    
    Returns:
        int: Required number of gradient accumulation steps
    """
    # Calculate the necessary gradient accumulation steps
    grad_accum_steps = total_tokens // (seq_len * batch_size)
    
    return max(1, grad_accum_steps)  # Ensure at least 1 accumulation step





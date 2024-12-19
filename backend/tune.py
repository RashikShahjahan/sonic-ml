import numpy as np
import psutil
import torch
from typing import Optional

def calculate_memory_per_token(max_seq_len: int, is_cpu: bool = False):
    """
    Calculate memory needed per token (used as a rough baseline).
    This does NOT fully account for the attention matrix (which is quadratic).
    We'll keep it for minor overhead estimates.
    """
    bytes_per_token = 4  # float32

    if is_cpu:
        # Very rough overhead factors
        attention_memory = max_seq_len
        projection_memory = 3
        ffn_memory = 4
        cpu_overhead = 2

        total_memory = (attention_memory + projection_memory + ffn_memory + cpu_overhead)
        # Add 20% fragmentation overhead
        return bytes_per_token * total_memory * 1.2
    else:
        # GPU side: smaller overhead factor
        activation_factor = 8
        attention_overhead = max_seq_len * 2
        return bytes_per_token * (activation_factor + attention_overhead)


def calculate_chunk_size(max_available_memory_gb: float,
                         batch_size: int,
                         max_seq_len: int,
                         safety_factor: float = 0.3) -> int:
    """
    Calculate an 'optimal' chunk size ensuring we can hold
    chunk data in memory. We rely on the simpler memory_per_token
    estimate here, but keep it conservative with the safety factor.
    """
    max_memory_bytes = max_available_memory_gb * safety_factor * 1024**3
    bytes_per_token = calculate_memory_per_token(max_seq_len)
    bytes_per_sample = max_seq_len * bytes_per_token

    chunk_size = int(max_memory_bytes / (bytes_per_sample * batch_size))
    chunk_size = max(chunk_size, 1)  # can't be zero
    # Round to nearest hundred for convenience
    chunk_size = (chunk_size // 100) * 100

    # Ensure chunk_size >= batch_size to avoid overhead complexity
    return max(batch_size, chunk_size)


def calculate_num_parameters(vocab_size: int, dim: int, n_layers: int,
                             n_heads: int, n_kv_heads: int = None,
                             hidden_dim: int = None, multiple_of: int = 256) -> int:
    """
    Calculate the total number of learnable parameters in the Transformer model.
    """
    n_params = 0

    # Token embeddings
    n_params += vocab_size * dim

    for _ in range(n_layers):
        n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        # q, k, v projections
        n_params += dim * (n_heads * (dim // n_heads))  # wq
        n_params += dim * (n_kv_heads * (dim // n_heads))  # wk
        n_params += dim * (n_kv_heads * (dim // n_heads))  # wv
        n_params += (n_heads * (dim // n_heads)) * dim  # wo
        n_params += dim  # attention_norm

        # FeedForward (approx 4x, or a custom hidden_dim)
        ff_hidden = 4 * dim
        ff_hidden = int(2 * ff_hidden / 3)
        ff_hidden = multiple_of * ((ff_hidden + multiple_of - 1) // multiple_of)
        if hidden_dim is not None:
            ff_hidden = hidden_dim

        n_params += dim * ff_hidden  # w1
        n_params += ff_hidden * dim  # w2
        n_params += dim * ff_hidden  # w3
        n_params += dim  # ffn_norm

    # final norm & output
    n_params += dim  # norm
    n_params += dim * vocab_size  # output layer

    return n_params


def calculate_model_memory(n_params: int, dtype_bytes: int = 4) -> float:
    """
    Calculate approximate memory usage for model weights, optimizer states, and gradients in GB.
    """
    # Model weights
    model_memory = n_params * dtype_bytes
    
    # Adam optimizer states (m and v)
    optimizer_memory = 2 * model_memory
    
    # Gradients
    gradient_memory = model_memory
    
    # Add 10% buffer for miscellaneous tensors
    misc_memory = (model_memory + optimizer_memory + gradient_memory) * 0.1
    
    total = model_memory + optimizer_memory + gradient_memory + misc_memory
    return total / (1024**3)


def calculate_max_batch_size(max_available_memory_gb: float,
                           max_seq_len: int,
                           model_memory_gb: float,
                           n_heads: int = 8,
                           n_layers: int = 5,
                           dim: int = 64,
                           safety_factor: float = 0.5,
                           is_cpu: bool = False) -> int:
    """
    More accurate memory estimation for finding maximum batch size
    """
    # Reserve overhead for CPU/system
    overhead_gb = 2.0 if is_cpu else 1.0
    usable_memory_gb = max(0, (max_available_memory_gb - overhead_gb - model_memory_gb) * safety_factor)

    def estimate_batch_memory(batch_size: int) -> float:
        """
        Estimate memory usage per batch:
        1. KV Cache: batch_size * seq_len * 2 * dim * n_layers
        2. Attention: batch_size * n_heads * seq_len * seq_len * n_layers
        3. Activations: batch_size * seq_len * dim * activation_factor
        4. Layer intermediates: batch_size * seq_len * dim * 4 (FFN expansion)
        """
        bytes_per_float = 4  # float32
        
        # KV Cache
        kv_cache_bytes = batch_size * max_seq_len * 2 * dim * n_layers * bytes_per_float
        
        # Attention matrices (Q @ K)
        attn_bytes = batch_size * n_heads * max_seq_len * max_seq_len * n_layers * bytes_per_float
        
        # Activations (roughly 4x the embedding dim for transformer blocks)
        activation_factor = 4
        activation_bytes = batch_size * max_seq_len * dim * activation_factor * bytes_per_float
        
        # FFN intermediates
        ffn_bytes = batch_size * max_seq_len * dim * 4 * bytes_per_float
        
        # Add 20% for temporary tensors and fragmentation
        total_bytes = (kv_cache_bytes + attn_bytes + activation_bytes + ffn_bytes) * 1.2
        
        return total_bytes / (1024**3)  # Convert to GB

    # Binary search for largest fitting batch size
    low, high = 1, 8192
    best = 1
    while low <= high:
        mid = (low + high) // 2
        used_gb = estimate_batch_memory(mid)
        if used_gb <= usable_memory_gb:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    # Round down to nearest power of 2 for better performance
    return 2 ** (best.bit_length() - 1)


def find_optimal_batch_config(total_tokens: int,
                              max_seq_len: int,
                              total_memory_gb: float,
                              model_memory_gb: float,
                              min_batch_size: int = 1,
                              is_cpu: bool = False,
                              n_heads: int = 8,
                              n_layers: int = 5,
                              dim: int = 64,
                              safety_factor: float = 0.5) -> tuple[int, int, int, float]:
    """
    Find an optimal batch size and gradient accumulation steps configuration
    that attempts to stay within the memory budget for CPU or GPU.
    """
    # 1) Find maximum feasible batch size via binary search
    max_batch = calculate_max_batch_size(
        max_available_memory_gb=total_memory_gb,
        max_seq_len=max_seq_len,
        model_memory_gb=model_memory_gb,
        n_heads=n_heads,
        n_layers=n_layers,
        dim=dim,
        safety_factor=safety_factor,
        is_cpu=is_cpu
    )
    # Ensure min_batch_size is a power of 2
    min_batch_size = 2 ** (min_batch_size - 1).bit_length()
    batch_size = max(min_batch_size, max_batch)

    # 2) Compute gradient accumulation steps to reach total_tokens
    tokens_per_batch = batch_size * max_seq_len
    grad_accum_steps = max(1, (total_tokens + tokens_per_batch - 1) // tokens_per_batch)

    # 3) Use constant chunk size of 1000
    chunk_size = 500

    # 4) Final memory usage with that batch size
    attn_bytes = (batch_size * max_seq_len * max_seq_len * n_heads * 4 * n_layers)
    attn_gb = attn_bytes / (1024**3)
    act_bytes = batch_size * max_seq_len * dim * 4 * 2
    act_gb = act_bytes / (1024**3)
    total_used_memory_gb = model_memory_gb + attn_gb + act_gb

    return batch_size, grad_accum_steps, chunk_size, total_used_memory_gb


def measure_memory_usage(is_cuda: bool = False) -> tuple[float, Optional[float]]:
    """
    Measure current memory usage for CPU and optionally GPU.
    Returns tuple of (cpu_gb, gpu_gb) where gpu_gb is None if CUDA not available.
    """
    # CPU Memory
    process = psutil.Process()
    cpu_gb = process.memory_info().rss / (1024 ** 3)
    
    # GPU Memory (if available)
    gpu_gb = None
    if is_cuda and torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    
    return cpu_gb, gpu_gb

def log_memory_usage(step: int, batch_size: int, max_seq_len: int, is_cuda: bool = False, log_file: str = "memory_usage.log"):
    """
    Log current memory usage statistics to a file.
    """
    cpu_gb, gpu_gb = measure_memory_usage(is_cuda)
    
    log_message = f"""
Memory Usage (Step {step}):
{'='*50}
Batch Size: {batch_size}, Sequence Length: {max_seq_len}
CPU Memory: {cpu_gb:.2f} GB
{f'GPU Memory: {gpu_gb:.2f} GB' if gpu_gb is not None else ''}
{'='*50}
"""
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(log_message)
    
    return cpu_gb, gpu_gb





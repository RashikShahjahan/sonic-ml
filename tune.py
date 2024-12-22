import numpy as np
import psutil
import torch



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

    total = model_memory + optimizer_memory + gradient_memory 
    return total / (1024**3)


def calculate_max_batch_size(max_available_memory_gb: float,
                             max_seq_len: int,
                             model_memory_gb: float,
                             n_heads:int,
                             n_layers: int,
                             dim: int,
                             ) -> int:
    """
    Calculate the maximum batch size that can be used without exceeding memory constraints.
    """

    # Memory left for training
    usable_memory_gb = 0.3*max_available_memory_gb

    def estimate_batch_memory(batch_size: int) -> float:
        bytes_per_float = 4

        # Model parameter bytes
        param_bytes = model_memory_gb * (1024 ** 3)

        # 1) Base input memory: depends on batch size & sequence length
        base_memory_bytes = batch_size * max_seq_len * bytes_per_float
        
        # 2) Activations (forward + backward): account for each layer
        activation_memory_bytes = (
            base_memory_bytes  # Base memory per token
            * dim             # Dimension of embeddings
            * n_layers       # Each layer produces activations
            * 2             # Forward + backward
        )
        
        # 3) Gradients: scales with param_bytes
        gradient_memory_bytes = param_bytes

        # 4) Optimizer states: typically 2x param bytes for Adam (momentum + variance)
        optimizer_memory_bytes = 2.0 * param_bytes


        # 5) Attention mechanism: intermediate activations for scaled dot-product attention
        attention_memory_bytes = (
            batch_size 
            * (max_seq_len ** 2)  # Sequence-to-sequence attention scores
            * n_heads 
            * n_layers 
            * bytes_per_float
        )

        total_bytes = (
            activation_memory_bytes
            + gradient_memory_bytes
            + optimizer_memory_bytes
            + attention_memory_bytes
        )

        return total_bytes / (1024 ** 3)  # Convert to GB
    
    for batch_size in [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        used_gb = estimate_batch_memory(batch_size)
        print(f"Batch size: {batch_size}, Used memory: {used_gb:.2f} GB")
        if used_gb <= usable_memory_gb:
            return batch_size


def find_optimal_batch_config(
    max_seq_len: int,
    model_memory_gb: float,
    target_batch_tokens: int = 100_000,
    is_cpu: bool = True,
    vocab_size: int = 50257,
    dim: int = 768,
    n_layers: int = 12,
    n_heads: int = 12
) -> None:
    """
    Find and print optimal batch configuration based on memory constraints and target batch tokens.
    """
    # Get available memory
    if is_cpu:
        total_memory = psutil.virtual_memory().available / (1024**3)  # Convert to GB
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but is_cpu=False")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Find maximum batch size that fits in memory
    max_batch_size = calculate_max_batch_size(
        max_available_memory_gb=total_memory,
        max_seq_len=max_seq_len,
        model_memory_gb=model_memory_gb,
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads
    )
    
    tokens_per_batch = max_batch_size * max_seq_len
    grad_accum = max(1, int(np.floor(target_batch_tokens / tokens_per_batch)))

    
    total_tokens = max_batch_size * max_seq_len * grad_accum
    
    print("\nDetailed Configuration Report:")
    print("=" * 50)
    print(f"System Configuration:")
    print(f"- Total available memory: {total_memory:.2f} GB")
    print(f"- Running on: {'CPU' if is_cpu else 'GPU'}")
    print(f"\nModel Parameters:")
    print(f"- Vocabulary size: {vocab_size:,}")
    print(f"- Model dimension: {dim}")
    print(f"- Number of layers: {n_layers}")
    print(f"- Number of heads: {n_heads}")
    print(f"\nTraining Configuration:")
    print(f"- Maximum sequence length: {max_seq_len}")
    print(f"- Target tokens per batch: {target_batch_tokens:,}")
    print(f"- Maximum gradient accumulation: {grad_accum}")
    print(f"\nCalculated Settings:")
    print(f"- Batch size: {max_batch_size}")
    print(f"- Gradient accumulation steps: {grad_accum}")
    print(f"- Tokens per batch: {tokens_per_batch:,}")
    print(f"- Total tokens per iteration: {total_tokens:,}")
    print("=" * 50)






# Model configuration
vocab_size = 4096
dim = 288
n_layers = 6
n_heads = 6




# Calculate training parameters
n_params = calculate_num_parameters(vocab_size, dim, n_layers, n_heads)
model_memory = calculate_model_memory(n_params)
print(f"Total parameters: {n_params:,}")
print(f"Model memory: {model_memory:.2f} GB")

# Training configuration
total_memory_gb = 16 # Your system's total memory
max_seq_len = 256
desired_tokens = 100000  
chunk_size = 512

find_optimal_batch_config(
    max_seq_len=max_seq_len,
    model_memory_gb=model_memory,
    target_batch_tokens=desired_tokens,
    is_cpu=True,
    vocab_size=vocab_size,
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads
) 



from flow import train_workflow, inference_workflow, train_vocab, download_workflow
from tune import calculate_num_parameters, calculate_model_memory, calculate_chunk_size, calculate_max_batch_size, calculate_grad_accum_steps, calculate_desired_batch_size
import os

os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("tokenizers", exist_ok=True)

# First download the dataset
download_workflow(
    dataset_name="wikimedia/wikipedia",
    dataset_data_dir="20231101.bn",
    output_dir="datasets/wikipedia_bn"
)

vocab_size = 8192
dim = 64
n_layers = 5
n_heads = 8

# Calculate training parameters
n_params = calculate_num_parameters(vocab_size, dim, n_layers, n_heads)
model_memory = calculate_model_memory(n_params)
print(f"Total parameters: {n_params:,}")
print(f"Model memory: {model_memory:.2f} GB")

max_available_memory_gb = 8  
max_seq_len = 512
initial_batch_size = 64
total_tokens = 100000  # Desired tokens per update
max_batch_size = calculate_max_batch_size(max_available_memory_gb, max_seq_len, model_memory)
desired_batch_size = calculate_desired_batch_size(total_tokens, max_seq_len)
batch_size = min(desired_batch_size, max_batch_size)
grad_accum_steps = calculate_grad_accum_steps(total_tokens, max_seq_len, batch_size)
chunk_size = calculate_chunk_size(max_available_memory_gb, batch_size, max_seq_len)

print(f"Using batch size: {batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Chunk size: {chunk_size}")
print(f"Tokens per update: {batch_size * max_seq_len * grad_accum_steps:,}")

# Train tokenizer using local dataset
train_vocab(
    vocab_size=vocab_size, 
    dataset_path="datasets/wikipedia_bn",
    model_prefix="wikipedia_bn_tokenizer",
    chunk_size=chunk_size
)

# Train model using local dataset
train_workflow(
    num_steps=1000,
    batch_size=batch_size,
    learning_rate=0.001,
    vocab_size=vocab_size,
    dataset_path="datasets/wikipedia_bn",
    model_id="wikipedia_bn_model",
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    max_seq_len=max_seq_len,
    tokenizer_prefix="wikipedia_bn_tokenizer",
    gradient_accumulation_steps=grad_accum_steps,
    chunk_size=chunk_size
)

# Inference remains the same
inference_workflow(
    model_path="checkpoints/wikipedia_bn_model_checkpoint_step_1000.pth",
    prompt="আপনি কেমন আছেন?",
    tokenizer_prefix="wikipedia_bn_tokenizer"
)

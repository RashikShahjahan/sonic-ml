from flow import train_workflow, inference_workflow, train_vocab, download_workflow
from tune import (
    calculate_num_parameters, 
    calculate_model_memory, 
    find_optimal_batch_config
)
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

# Model configuration
vocab_size = 8192

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

batch_size, grad_accum_steps, total_used_memory_gb = find_optimal_batch_config(
    total_tokens=desired_tokens,
    max_seq_len=max_seq_len,
    total_memory_gb=total_memory_gb,
    model_memory_gb=model_memory,
    min_batch_size=1,
    is_cpu=True,  
    n_heads=n_heads,
    n_layers=n_layers,
    dim=dim,
    safety_factor=0.7 
) 

actual_tokens = batch_size * max_seq_len * grad_accum_steps

print("\nTraining Configuration:")
print(f"{'='*50}")
print(f"Model Architecture:")
print(f"  Vocabulary Size: {vocab_size:,}")
print(f"  Dimension: {dim}")
print(f"  Layers: {n_layers}")
print(f"  Attention Heads: {n_heads}")
print(f"\nMemory Usage:")
print(f"  Total Available Memory: {total_memory_gb} GB")
print(f"  Model Memory: {model_memory:.2f} GB")
print(f"  Used Memory: {total_used_memory_gb:.2f} GB")
print(f"\nBatch Configuration:")
print(f"  Sequence Length: {max_seq_len}")
print(f"  Batch Size: {batch_size}")
print(f"  Gradient Accumulation Steps: {grad_accum_steps}")
print(f"  Chunk Size: {chunk_size}")
print(f"  Tokens per Update: {actual_tokens:,} (target: {desired_tokens:,})")
print(f"{'='*50}\n")

input("Review the configuration above. Press Enter to continue or Ctrl+C to abort: ")


# Train tokenizer using local dataset
"""
train_vocab(
    vocab_size=vocab_size, 
    dataset_path="datasets/wikipedia_bn",
    model_prefix="wikipedia_bn_tokenizer",
    chunk_size=chunk_size
)
"""

# Train model using local dataset
train_workflow(
    num_steps=6000,
    batch_size=batch_size,
    learning_rate=0.0005,
    vocab_size=vocab_size,
    dataset_path="datasets/wikipedia_bn",
    model_id="wikipedia_bn_model_15M",
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    max_seq_len=max_seq_len,
    tokenizer_prefix="wikipedia_bn_tokenizer",
    gradient_accumulation_steps=grad_accum_steps,
    chunk_size=chunk_size,
    resume_from_checkpoint=True

)
# Test inference with Wikipedia-style prompts
test_prompts = [
    "বাংলাদেশের ইতিহাস",     # History of Bangladesh
    "রবীন্দ্রনাথ ঠাকুর হলেন", # Rabindranath Tagore is
    "ঢাকা শহর হল",           # Dhaka city is
    "সুন্দরবন হল একটি",      # The Sundarbans is a
]

print("\nTesting inference with Wikipedia-style prompts:")
print("="*50)
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    inference_workflow(
        model_path="checkpoints/wikipedia_bn_model_15M_checkpoint_step_99.pth",
        prompt=prompt,
        tokenizer_prefix="wikipedia_bn_tokenizer"
    )
    print("-"*50)




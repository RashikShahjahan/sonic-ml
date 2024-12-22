from flow import train_workflow, inference_workflow, train_vocab, download_workflow

import os

os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("tokenizers", exist_ok=True)
"""

# First download the dataset
download_workflow(
    dataset_name="wikimedia/wikipedia",
    dataset_data_dir="20231101.bn",
    output_dir="datasets/wikipedia_bn"
)



# Train tokenizer using local dataset
train_vocab(
    vocab_size=4096, 
    dataset_path="datasets/wikipedia_bn",
    model_prefix="wikipedia_bn_tokenizer_4096",
    chunk_size=512
)
"""

# Train model using local dataset
train_workflow(
    num_steps=18000,
    batch_size=64,
    learning_rate=0.0005,
    vocab_size=4096,
    dataset_path="datasets/wikipedia_bn",
    model_id="wikipedia_bn_model_15M_4096",
    dim=288,
    n_layers=6,
    n_heads=6,
    max_seq_len=256,
    tokenizer_prefix="wikipedia_bn_tokenizer_4096",
    gradient_accumulation_steps=8,
    chunk_size=512,
    resume_from_checkpoint=True
)




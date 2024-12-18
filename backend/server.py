from flow import train_workflow, inference_workflow, train_vocab, download_workflow

import os
os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("tokenizers", exist_ok=True)

# First download the dataset
download_workflow(
    dataset_name="ai4bharat/sangraha",
    dataset_data_dir="verified/ben",
    output_dir="datasets/sangraha_ben"
)



# Train tokenizer using local dataset
train_vocab(
    vocab_size=4096, 
    dataset_path="datasets/sangraha_ben",
    model_prefix="bangla_tokenizer"
)

# Train model using local dataset
train_workflow(
    num_steps=1000,
    batch_size=64,
    learning_rate=0.001,
    vocab_size=4096,
    dataset_path="datasets/sangraha_ben",
    model_id="bangla_model",
    dim=64,
    n_layers=5,
    n_heads=8,
    max_seq_len=512,
    tokenizer_model="bangla_tokenizer",
    gradient_accumulation_steps=4
)

# Inference remains the same
inference_workflow(
    model_path="checkpoints/bangla_model_checkpoint_step_999.pth",
    prompt="আপনি কেমন আছেন?",
    tokenizer_model="bangla_tokenizer.model"
)

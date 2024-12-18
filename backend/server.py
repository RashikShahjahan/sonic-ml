from flow import train_workflow, inference_workflow, train_vocab
"""
train_vocab(
    vocab_size=4096, 
    dataset_name="ai4bharat/sangraha", 
    dataset_data_dir="verified/ben", 
    dataset_size=10000,
    model_prefix="bangla_tokenizer"
)
"""

train_workflow(
    num_steps=1000,
    batch_size=64,
    learning_rate=0.001,
    vocab_size=4096,
    dataset_name="ai4bharat/sangraha",
    dataset_data_dir="verified/ben",
    dataset_size=10000,
    model_id="bangla_model",
    dim=64,
    n_layers=5,
    n_heads=8,
    max_seq_len=512,
    tokenizer_model="bangla_tokenizer.model",
    gradient_accumulation_steps=4
)




inference_workflow(
    model_path="checkpoints/bangla_model_checkpoint_step_999.pth",
    prompt="আপনি কেমন আছেন?",
    tokenizer_model="bangla_tokenizer.model"
)

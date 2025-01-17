from celery import Celery
from sonic_ml.commands.train_vocab import train_vocab
from sonic_ml.commands.train_model import train
from sonic_ml.commands.download_data import download_workflow
from sonic_ml.commands.eval_model import inference_workflow

# Initialize Celery
app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def download_data_task(dataset_name, dataset_data_dir, output_dir):
    return download_workflow(
        dataset_name=dataset_name,
        dataset_data_dir=dataset_data_dir,
        output_dir=output_dir
    )

@app.task
def train_vocab_task(vocab_size, dataset_path, model_prefix, chunk_size):
    return train_vocab(
        vocab_size=vocab_size,
        dataset_path=dataset_path,
        model_prefix=model_prefix,
        chunk_size=chunk_size
    )

@app.task
def train_model_task(num_steps, batch_size, learning_rate, vocab_size, dataset_path,
                     dim, n_layers, n_heads, max_seq_len, tokenizer_prefix,
                     gradient_accumulation_steps, chunk_size, resume_from_checkpoint,
                     model_id, model_architecture):
    return train(
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        vocab_size=vocab_size,
        dataset_path=dataset_path,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        tokenizer_prefix=tokenizer_prefix,
        gradient_accumulation_steps=gradient_accumulation_steps,
        chunk_size=chunk_size,
        resume_from_checkpoint=resume_from_checkpoint,
        model_id=model_id,
        model_architecture=model_architecture
    )

@app.task
def eval_model_task(model_id, prompt, max_new_tokens, temperature, top_k, tokenizer_prefix):
    return inference_workflow(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        tokenizer_prefix=tokenizer_prefix
    )


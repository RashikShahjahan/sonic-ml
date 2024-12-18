import torch
from datasets import load_dataset, Dataset
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Transformer, ModelArgs
from tokenizer import  Tokenizer
from utils import save_checkpoint
from tokenizer import Tokenizer
import os
import sentencepiece as spm
from typing import List
from utils import load_model
from flytekit import task, workflow


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
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_dataloader



@task
def train(num_steps: int, learning_rate: float, dim: int, n_layers: int, n_heads: int, 
          vocab_size: int, max_seq_len: int, dataset: Dataset, model_id: str, 
          gradient_accumulation_steps: int, dataset_path: str, chunk_indices: List[int],
          batch_size: int, tokenizer_prefix: str):
    """Train a Transformer model on chunked dataset with gradient accumulation.
    
    Args:
        num_steps (int): Total number of training steps
        learning_rate (float): Learning rate for the optimizer
        dim (int): Model dimension/embedding size
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        vocab_size (int): Size of the vocabulary
        max_seq_len (int): Maximum sequence length for input texts
        dataset (Dataset): Initial chunk of the dataset to start training
        model_id (str): Unique identifier for the model (used in checkpoint names)
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        dataset_path (str): Path to the full dataset on disk
        chunk_indices (List[int]): Starting indices for each data chunk
        batch_size (int): Number of samples per batch
        tokenizer_prefix (str): Prefix for the tokenizer model file
    
    Returns:
        Transformer: The trained model
        
    The function implements the following training loop:
    1. Initializes model, optimizer, and loads first data chunk
    2. For each step:
        - Fetches next batch or loads new chunk if needed
        - Performs forward and backward passes
        - Updates model weights every gradient_accumulation_steps
        - Saves checkpoints every 100 steps
    3. Uses gradient accumulation for effective larger batch sizes
    4. Processes dataset in chunks to handle large datasets
    """
    model_args = ModelArgs(
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        vocab_size=vocab_size, max_seq_len=max_seq_len,
    )
    model = Transformer(model_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95),device_type=device)
    model.to(device)

    # Initial preprocessing of first chunk
    train_dataloader = preprocess_dataset(
        dataset=dataset,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tokenizer_prefix=tokenizer_prefix
    )
    
    dataloader_iterator = iter(train_dataloader)
    current_chunk_idx = 1

    # Training loop
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(num_steps), desc="Training")
    
    optimizer.zero_grad()
    
    for step in progress_bar:
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            if current_chunk_idx < len(chunk_indices):
                start_idx = chunk_indices[current_chunk_idx]
                end_idx = start_idx + batch_size
                
                full_dataset = datasets.load_from_disk(dataset_path)
                if isinstance(full_dataset, datasets.DatasetDict):
                    split_name = 'train' if 'train' in full_dataset else list(full_dataset.keys())[0]
                    full_dataset = full_dataset[split_name]

                current_chunk = full_dataset.select(range(start_idx, end_idx))
                dataloader_iterator = iter(preprocess_dataset(dataset=current_chunk, batch_size=batch_size, max_seq_len=max_seq_len, tokenizer_prefix=tokenizer_prefix))
                current_chunk_idx += 1
                print(f"\nLoading chunk {current_chunk_idx} of {len(chunk_indices)}")
            
            batch = next(dataloader_iterator)
        
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        model(input_ids, targets=targets)
        loss = model.last_loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps
        
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f"Step {step + 1}, Average loss: {avg_loss}")
            total_loss = 0
            save_checkpoint(
                model, optimizer, step, loss.item(),
                f'checkpoints/{model_id}_checkpoint_step_{step}.pth',
            )

        save_checkpoint(
            model, optimizer, step, loss.item(),
            f'checkpoints/{model_id}_checkpoint_step_{step}.pth',
        )

    return model

@task
def generate_text(
    model_path: str,
    tokenizer_prefix: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text continuation from a given prompt using a trained transformer model.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        tokenizer_prefix (str): Prefix for the tokenizer model file
        prompt (str): Input text to continue from
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
        temperature (float, optional): Controls randomness in generation. Higher values (e.g., 1.0)
            make the output more random, lower values (e.g., 0.2) make it more deterministic. 
            Defaults to 0.8.
        top_k (int, optional): Number of highest probability vocabulary tokens to keep for 
            top-k-filtering. Defaults to 200.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
            Defaults to CUDA if available, else CPU.
    
    Returns:
        str: The generated text including the original prompt
        
    The function performs the following steps:
    1. Tokenizes the input prompt
    2. Loads the model from the checkpoint
    3. Generates new tokens using the model
    4. Decodes the generated tokens back to text
    """
    # Construct full path to tokenizer model
    tokenizer_path = os.path.join("tokenizers", f"{tokenizer_prefix}.model")
    tokenizer = Tokenizer(tokenizer_path)
    # Encode the prompt
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(encoded).unsqueeze(0).to(device)  # Create a batch of size 1
    
    # Move model to device
    model = load_model(model_path).to(device)
    
    # Generate tokens
    with torch.no_grad():
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    return generated_text

@task
def download_dataset(dataset_name: str, dataset_data_dir: str, output_dir: str) -> str:
    """Download a dataset from Hugging Face and save it locally.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., 'wikipedia', 'wikitext')
        dataset_data_dir (str): Directory containing additional data files required by the dataset
        output_dir (str): Local directory path where the dataset will be saved
    
    Returns:
        str: Path to the directory where the dataset was saved
        
    The function performs the following steps:
    1. Checks if the dataset is already downloaded, if not downloads the specified dataset from Hugging Face using the datasets library
    2. Saves the entire dataset to disk in the specified output directory
    3. Returns the path to the saved dataset
    
    Example:
        >>> output_path = download_dataset(
        ...     dataset_name='wikitext',
        ...     dataset_data_dir='data/raw',
        ...     output_dir='data/processed'
        ... )
    """
    if os.path.exists(output_dir):
        print(f"Dataset already downloaded at {output_dir}")
        return output_dir
    
    dataset = load_dataset(dataset_name, data_dir=dataset_data_dir)
    
    dataset.save_to_disk(output_dir)
    return output_dir

@task
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


@task
def train_tokenizer(dataset: Dataset, vocab_size: int, model_prefix: str = "tokenizer") -> None:
    """Train a SentencePiece tokenizer on the provided dataset.
    
    Args:
        dataset (Dataset): The dataset containing text samples in a 'text' column
        vocab_size (int): Size of the vocabulary to generate
        model_prefix (str, optional): Prefix for the output model files. Defaults to "tokenizer"
            Will create {model_prefix}.model and {model_prefix}.vocab files
    
    The function performs the following steps:
    1. Writes all text samples to a temporary file
    2. Trains a BPE tokenizer using SentencePiece with the following settings:
        - BPE (Byte-Pair Encoding) model type
        - Full character coverage
        - Special token IDs: PAD=3, BOS=1, EOS=2, UNK=0
        - Whitespace preservation and digit splitting enabled
        - Byte fallback for unknown characters
    3. Cleans up the temporary training file
    
    The resulting tokenizer files will be saved in the 'tokenizers/' directory.
    """
    # Create a temporary file to write the text data
    with open("temp_train_data.txt", "w", encoding="utf-8") as f:
        for text in dataset['text']:
            f.write(text + "\n")
    
    # Train the tokenizer using the temporary file
    spm.SentencePieceTrainer.train(
        input="temp_train_data.txt",
        model_prefix=f"tokenizers/{model_prefix}",
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_piece="[PAD]",
    )
    
    os.remove("temp_train_data.txt")

@workflow
def download_workflow(dataset_name: str, dataset_data_dir: str, output_dir: str):
    """Workflow for downloading and saving a dataset from Hugging Face Hub locally.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., 'wikipedia', 'wikitext')
        dataset_data_dir (str): Directory containing additional data files required by the dataset
        output_dir (str): Local directory path where the dataset will be saved

    """
    download_dataset(dataset_name=dataset_name, dataset_data_dir=dataset_data_dir, output_dir=output_dir)



@workflow
def train_workflow(num_steps: int, batch_size: int, learning_rate: float, vocab_size: int,
                  dim: int, n_layers: int, n_heads: int, max_seq_len: int, tokenizer_prefix: str, 
                  model_id: str, dataset_path: str, gradient_accumulation_steps: int, chunk_size: int):
    """Orchestrates the complete training workflow for the transformer model.
    
    Args:
        num_steps (int): Total number of training steps
        batch_size (int): Number of samples per batch
        learning_rate (float): Learning rate for the optimizer
        vocab_size (int): Size of the vocabulary
        dim (int): Model dimension/embedding size
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        max_seq_len (int): Maximum sequence length for input texts
        tokenizer_prefix (str): Prefix for the tokenizer model file
        model_id (str): Unique identifier for the model (used in checkpoint names)
        dataset_path (str): Path to the dataset on disk
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        chunk_size (int): Number of samples per chunk
    """
    # Load the initial chunk of the dataset
    current_chunk, chunk_indices = load_and_prepare_dataset(dataset_path=dataset_path, chunk_size=chunk_size)

    
    train(
        num_steps=num_steps, 
        learning_rate=learning_rate, 
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dataset=current_chunk,
        model_id=model_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_path=dataset_path,
        chunk_indices=chunk_indices,
        batch_size=batch_size,
        tokenizer_prefix=tokenizer_prefix
    )

@workflow
def train_vocab(vocab_size: int, dataset_path: str, model_prefix: str = "tokenizer", chunk_size: int = 1000):
    """Workflow for training a SentencePiece tokenizer on a dataset.
    
    Args:
        vocab_size (int): Size of the vocabulary to generate
        dataset_path (str): Path to the saved dataset on disk
        model_prefix (str, optional): Prefix for the output model files. Defaults to "tokenizer".
            Will create {model_prefix}.model and {model_prefix}.vocab files
    
    Returns:
        None: The tokenizer files are saved to disk in the 'tokenizers/' directory
    """
    dataset, _ = load_and_prepare_dataset(dataset_path=dataset_path, chunk_size=chunk_size)
    
    return train_tokenizer(dataset=dataset, vocab_size=vocab_size, model_prefix=model_prefix)



@workflow
def inference_workflow(model_path: str, tokenizer_prefix: str, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 200):
    """Generate text using a trained model.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        tokenizer_prefix (str): Prefix for the tokenizer model file (without .model extension)
        prompt (str): Input text to continue from
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
        temperature (float, optional): Controls randomness in generation. Defaults to 0.8.
        top_k (int, optional): Number of highest probability tokens to keep. Defaults to 200.
    """
    text = generate_text(
        model_path=model_path, 
        tokenizer_prefix=tokenizer_prefix,  # Pass the prefix only, not the full path
        prompt=prompt, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    )
    print(text)


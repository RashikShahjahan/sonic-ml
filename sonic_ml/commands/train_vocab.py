import os
import sentencepiece as spm
from sonic_ml.utils.core import load_and_prepare_dataset

def train_tokenizer(dataset_path: str, vocab_size: int, chunk_size: int, model_prefix: str = "tokenizer") -> None:
    """Train a SentencePiece tokenizer on the provided dataset.
    
    Args:
        dataset_path (str): Path to the dataset on disk
        vocab_size (int): Size of the vocabulary to generate
        chunk_size (int): Size of chunks to process
        model_prefix (str, optional): Prefix for the output model files. Defaults to "tokenizer"
            Will create {model_prefix}.model and {model_prefix}.vocab files
    """
    # Move dataset loading into the task
    dataset, _ = load_and_prepare_dataset(dataset_path=dataset_path, chunk_size=chunk_size)
    
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

def train_vocab(vocab_size: int, dataset_path: str, model_prefix: str = "tokenizer", chunk_size: int = 1000):
    """Workflow for training a SentencePiece tokenizer on a dataset.
    
    Args:
        vocab_size (int): Size of the vocabulary to generate
        dataset_path (str): Path to the saved dataset on disk
        model_prefix (str, optional): Prefix for the output model files. Defaults to "tokenizer"
        chunk_size (int, optional): Size of chunks to process. Defaults to 1000
    
    Returns:
        None: The tokenizer files are saved to disk in the 'tokenizers/' directory
    """
    return train_tokenizer(
        dataset_path=dataset_path,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        model_prefix=model_prefix
    )
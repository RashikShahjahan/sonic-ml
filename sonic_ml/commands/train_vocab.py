import os
import sentencepiece as spm
from datasets import Dataset
from sonic_ml.utils.utils import load_and_prepare_dataset
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
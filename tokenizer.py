# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
from datasets import Dataset
from typing import List

from sentencepiece import SentencePieceProcessor
import sentencepiece as spm


TOKENIZER_MODEL = "tokenizer.model" # the llama sentencepiece tokenizer model

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        
        # Print these values for debugging
        print(f"Tokenizer initialized with:")
        print(f"- vocab_size: {self.n_words}")
        print(f"- bos_id: {self.bos_id}")
        print(f"- eos_id: {self.eos_id}")
        print(f"- pad_id: {self.pad_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors="pt"):
        """Add padding method required by HuggingFace's data collator"""
        import torch
        
        if not padding:
            return encoded_inputs
            
        # Find max length in batch if max_length not specified
        if max_length is None:
            max_length = max(len(x) for x in encoded_inputs["input_ids"])
            
        # Always use 0 as padding token to avoid negative indices
        pad_token = 0
        
        padded_inputs = []
        attention_mask = []
        
        for ids in encoded_inputs["input_ids"]:
            padding_length = max_length - len(ids)
            padded_inputs.append(ids + [pad_token] * padding_length)
            attention_mask.append([1] * len(ids) + [0] * padding_length)
            
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_inputs),
                "attention_mask": torch.tensor(attention_mask)
            }
        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask
        }

def train_vocab(vocab_size: int, dataset: Dataset):
    # Convert dataset to list of strings
    train_text = dataset["train"]['text']
    
    # Create a temporary file to write the text data
    with open("temp_train_data.txt", "w", encoding="utf-8") as f:
        for text in train_text:
            f.write(text + "\n")
    
    # Train the tokenizer using the temporary file
    spm.SentencePieceTrainer.train(
        input="temp_train_data.txt",
        model_prefix="tokenizer",
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
    
    # Optionally, clean up the temporary file
    os.remove("temp_train_data.txt")
    


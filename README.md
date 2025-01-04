# Sonic ML

A command-line interface (CLI) tool for training and evaluating language models. It's an easy way to train tiny language models on your personal CPU, GPU or MacBook

## Features

-  Download datasets from Hugging Face Hub
-  Train custom tokenizers using SentencePiece
-  Train transformer models with configurable architectures(Currently supports LLaMA)
-  Evaluate models with customizable generation parameters

## Usage

### 1. Download a Dataset

```bash
sonic download_data --dataset_name "tiny_shakespeare"
```
or
```bash
sonic --config example_configs/download_data.yml download_data
```

### 2. Train a Tokenizer

```bash
sonic train_vocab \
--dataset "roneneldan/TinyStories" \
--vocab_size 4096 \
--model_id "TinyStories_4096" \
--chunk_size 512
```
or
```bash
sonic --config example_configs/train_vocab.yml train_vocab
```

### 3. Train the Model

```bash
sonic train_model \
--dataset "roneneldan/TinyStories" \
--model_id "TinyStories_4096" \
--tokenizer_prefix "TinyStories_4096" \
--vocab_size 4096 \
--dim 288 \
--n_layers 6 \
--n_heads 6 \
--max_seq_len 256 \
--batch_size 64 \
--steps 10000 \
--learning_rate 0.0005 \
--gradient_accumulation_steps 8 \
--chunk_size 512 \
--model_architecture llama
```
or
```bash
sonic --config example_configs/train_model.yml train_model
```

### 4. Evaluate the Model

```bash
sonic eval_model \
--model_id "TinyStories_4096" \
--tokenizer_prefix "TinyStories_4096" \
--prompt "My name is" \
--max_new_tokens 100 \
--temperature 0.8 \
--top_k 200
```
or
```bash
sonic --config example_configs/eval_model.yml eval_model
```

## License

MIT License

## Author

Rashik Shahjahan - https://www.rashik.sh/
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Look at the issues in the repo for ideas. Please include documentation on any tests that you have ran.



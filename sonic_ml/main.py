import argparse
import os
import yaml
from sonic_ml.commands.train_vocab import train_vocab
from sonic_ml.commands.train_model import train_workflow
from sonic_ml.commands.download_data import download_workflow
from sonic_ml.commands.eval_model import inference_workflow

def setup_dirs():
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)

def download_data(args):
    download_workflow(
        dataset_name=args.dataset_name,
        dataset_data_dir=args.dataset_data_dir,
        output_dir=f"datasets/{args.dataset_name}"
    )

def train_vocabulary(args):
    train_vocab(
        vocab_size=args.vocab_size,
        dataset_path=f"datasets/{args.dataset}",
        model_prefix=f"{args.model_id}",
        chunk_size=args.chunk_size
    )

def train_model(args):
    train_workflow(
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        vocab_size=args.vocab_size,
        dataset_path=f"datasets/{args.dataset}",
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        tokenizer_prefix=args.tokenizer_prefix,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chunk_size=args.chunk_size,
        resume_from_checkpoint=args.resume,
        model_id=args.model_id
    )


def eval(args):
    print(inference_workflow(
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        tokenizer_prefix=args.tokenizer_prefix
    ))

def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    if not config_path:
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(yaml_config, cli_args):
    """Merge YAML config with CLI arguments. CLI args take precedence."""
    config = yaml_config.copy()
    # Convert CLI args to dict, excluding None values
    cli_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
    config.update(cli_dict)
    return argparse.Namespace(**config)

def main():
    parser = argparse.ArgumentParser(description='Sonic ML CLI')
    parser.add_argument('--config', help='Path to YAML config file')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Download command
    download_parser = subparsers.add_parser('download_data', help='Download dataset')
    download_parser.add_argument('--dataset_name', required=True, help='Name of the dataset to download')
    download_parser.add_argument('--dataset_data_dir', default="", help='Path to the dataset data directory')


    # Train command
    train_parser = subparsers.add_parser('train_model', help='Train model')
    train_parser.add_argument('--dataset', required=True, help='Dataset name')
    train_parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size')
    train_parser.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    train_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    train_parser.add_argument('--dim', type=int, default=288, help='Model dimension')
    train_parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    train_parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    train_parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    train_parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    train_parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    train_parser.add_argument('--model_id', required=True, help='Model ID')
    train_parser.add_argument('--tokenizer_prefix', required=True, help='Tokenizer prefix')


    # Train vocab command
    train_vocab_parser = subparsers.add_parser('train_vocab', help='Train vocab')
    train_vocab_parser.add_argument('--dataset', required=True, help='Dataset name')
    train_vocab_parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size')
    train_vocab_parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size')
    train_vocab_parser.add_argument('--model_id', required=True, help='Model ID')

    # Eval command
    eval_parser = subparsers.add_parser('eval_model', help='Evaluate model')
    eval_parser.add_argument('--model_id', required=True, help='Model ID')
    eval_parser.add_argument('--prompt', required=True, help='Input prompt for inference')
    eval_parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum new tokens to generate')
    eval_parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for generation')
    eval_parser.add_argument('--top_k', type=int, default=200, help='Top-k for generation')
    eval_parser.add_argument('--tokenizer_prefix', required=True, help='Tokenizer prefix')



    args = parser.parse_args()
    setup_dirs()

    # Load YAML config if provided
    yaml_config = load_yaml_config(args.config)
    
    # Merge YAML config with CLI args
    if yaml_config:
        command_config = yaml_config.get(args.command, {})
        args = merge_configs(command_config, args)

    if args.command == 'download_data':
        download_data(args)
    elif args.command == 'train_vocab':
        train_vocabulary(args)
    elif args.command == 'train_model':
        train_model(args)
    elif args.command == 'eval_model':
        eval(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()




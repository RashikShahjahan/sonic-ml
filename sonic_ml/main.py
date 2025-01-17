import argparse
import os
import yaml
import sys
from celery.result import AsyncResult
from celery.app.control import Inspect
from sonic_ml.tasks import (
    download_data_task,
    train_vocab_task,
    train_model_task,
    eval_model_task,
    app  # Import the Celery app instance
)

def setup_dirs():
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)


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

def list_all_tasks():
    """List all active tasks and their status"""
    inspector = Inspect(app=app)
    
    # Get tasks from different states
    active = inspector.active() or {}
    reserved = inspector.reserved() or {}
    scheduled = inspector.scheduled() or {}
    
    if not any([active, reserved, scheduled]):
        print("No active tasks found")
        print("\nTo see historical tasks, you can still use Flower:")
        print("Run 'celery -A sonic_ml.tasks flower' to start the monitoring interface")
        return

    def print_tasks(task_dict, state):
        for worker, tasks in task_dict.items():
            if tasks:
                print(f"\n{state} tasks on {worker}:")
                for task in tasks:
                    task_id = task.get('id', 'Unknown')
                    name = task.get('name', 'Unknown')
                    args = task.get('args', [])
                    kwargs = task.get('kwargs', {})
                    print(f"  - ID: {task_id}")
                    print(f"    Command: {name}")
                    if args or kwargs:
                        print(f"    Arguments: {args} {kwargs}")
                    print()

    if active:
        print_tasks(active, "Currently running")
    if reserved:
        print_tasks(reserved, "Reserved")
    if scheduled:
        print_tasks(scheduled, "Scheduled")

def get_task_status(task_id):
    """Get status of a specific task"""
    result = AsyncResult(task_id)
    status = result.status
    if status == 'FAILURE':
        error = str(result.result)  # Get the error message
        return f"Task {task_id}: {status}\nError: {error}"
    return f"Task {task_id}: {status}"

def main():
    parser = argparse.ArgumentParser(description='Sonic ML CLI')
    parser.add_argument('--config', help='Path to YAML config file')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Download command
    download_parser = subparsers.add_parser('download_data', help='Download dataset')
    download_parser.add_argument('--dataset_name', required='--config' not in sys.argv, help='Name of the dataset to download')
    download_parser.add_argument('--dataset_data_dir', help='Path to the dataset data directory')


    # Train command
    train_parser = subparsers.add_parser('train_model', help='Train model')
    train_parser.add_argument('--dataset', required='--config' not in sys.argv, help='Dataset name')
    train_parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
    train_parser.add_argument('--steps', type=int,  help='Number of training steps')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--dim', type=int, help='Model dimension')
    train_parser.add_argument('--n_layers', type=int, help='Number of layers')
    train_parser.add_argument('--n_heads', type=int, help='Number of attention heads')
    train_parser.add_argument('--max_seq_len', type=int, help='Maximum sequence length')
    train_parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    train_parser.add_argument('--chunk_size', type=int, help='Chunk size')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    train_parser.add_argument('--model_id', required='--config' not in sys.argv, help='Model ID')
    train_parser.add_argument('--tokenizer_prefix', required='--config' not in sys.argv, help='Tokenizer prefix')
    train_parser.add_argument('--model_architecture', help='Model architecture')


    # Train vocab command
    train_vocab_parser = subparsers.add_parser('train_vocab', help='Train vocab')
    train_vocab_parser.add_argument('--dataset', required='--config' not in sys.argv, help='Dataset name')
    train_vocab_parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
    train_vocab_parser.add_argument('--chunk_size', type=int, help='Chunk size')
    train_vocab_parser.add_argument('--model_id', required='--config' not in sys.argv, help='Model ID')

    # Eval command
    eval_parser = subparsers.add_parser('eval_model', help='Evaluate model')
    eval_parser.add_argument('--model_id', required='--config' not in sys.argv, help='Model ID')
    eval_parser.add_argument('--prompt', required='--config' not in sys.argv, help='Input prompt for inference')
    eval_parser.add_argument('--max_new_tokens', type=int, help='Maximum new tokens to generate')
    eval_parser.add_argument('--temperature', type=float, help='Temperature for generation')
    eval_parser.add_argument('--top_k', type=int, help='Top-k for generation')
    eval_parser.add_argument('--tokenizer_prefix', required='--config' not in sys.argv, help='Tokenizer prefix')

    # Add status command
    status_parser = subparsers.add_parser('status', help='Check task status')
    status_parser.add_argument('task_id', help='Task ID to check')

    subparsers.add_parser('list', help='List all tasks')

    args = parser.parse_args()
    setup_dirs()

    # Load YAML config if provided
    yaml_config = load_yaml_config(args.config)
    
    # Merge YAML config with CLI args
    if yaml_config:
        command_config = yaml_config.get(args.command, {})
        args = merge_configs(command_config, args)

    if args.command == 'list':
        list_all_tasks()
        return
    elif args.command == 'status':
        print(get_task_status(args.task_id))
        return

    # For other commands, start them as Celery tasks
    if args.command in ['download_data', 'train_vocab', 'train_model', 'eval_model']:
        try:
            if args.command == 'download_data':
                task = download_data_task.delay(
                    dataset_name=args.dataset_name,
                    dataset_data_dir=args.dataset_data_dir,
                    output_dir=f"datasets/{args.dataset_name}"
                )
            elif args.command == 'train_vocab':
                task = train_vocab_task.delay(
                    vocab_size=args.vocab_size,
                    dataset_path=f"datasets/{args.dataset}",
                    model_prefix=f"{args.model_id}",
                    chunk_size=args.chunk_size
                )
            elif args.command == 'train_model':
                task = train_model_task.delay(
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
                    model_id=args.model_id,
                    model_architecture=args.model_architecture
                )
            elif args.command == 'eval_model':
                task = eval_model_task.delay(
                    model_id=args.model_id,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    tokenizer_prefix=args.tokenizer_prefix
                )
            
            print(f"Started {args.command} in background with task ID: {task.id}")
            
        except Exception as e:
            print(f"Task failed: {str(e)}")
            raise
    else:
        parser.print_help()

if __name__ == '__main__':
    main()




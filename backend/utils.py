import torch
from model import Transformer
from model import ModelArgs


# -----------------------------------------------------------------------------
# sampling utils

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def save_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer, step: int, loss: float, path: str):
    """Save model checkpoint including model arguments for complete restoration"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_args': model.params.__dict__,  # Save model arguments as dictionary
    }, path)



def load_model(model_path: str) -> Transformer:
    """
    Load a trained transformer model from a given path.
    
    Args:
        model_path: Path to the saved model state dict
        
    Returns:
        Loaded Transformer model
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    print(checkpoint.keys())
    
    # Initialize model with saved args
    model_args = ModelArgs(**checkpoint['model_args'])
    model = Transformer(model_args)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model



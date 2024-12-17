import torch
from model import Transformer
from tokenizer import Tokenizer
from model import ModelArgs
from typing import List
from tokenizer import Tokenizer

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

def save_checkpoint(model: Transformer, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)



def load_model(checkpoint_path: str, model_args: ModelArgs) -> Transformer:
    """Load the trained model from a checkpoint."""
    model = Transformer(model_args)
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Add weights_only=True for security
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model



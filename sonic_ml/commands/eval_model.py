import torch
import os
from sonic_ml.tokenizer.tokenizer import  Tokenizer
from sonic_ml.utils.utils import load_model
from flytekit import task, workflow

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
def get_latest_checkpoint(model_id: str) -> str:
    """Get the path to the latest checkpoint for a given model ID."""
    checkpoint_dir = os.path.join('checkpoints', model_id)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    # Get the checkpoint with the highest step number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_step_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

@workflow
def inference_workflow(model_id: str, tokenizer_prefix: str, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 200) -> str:
    """Generate text using a trained model."""
    checkpoint_path = get_latest_checkpoint(model_id=model_id)
    
    text = generate_text(
        model_path=checkpoint_path, 
        tokenizer_prefix=tokenizer_prefix,  
        prompt=prompt, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    )

    return text
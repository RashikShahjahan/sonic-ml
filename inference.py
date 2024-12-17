import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

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

def generate_text(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text from a prompt."""
    # Encode the prompt
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor(encoded).unsqueeze(0).to(device)  # Create a batch of size 1
    
    # Move model to device
    model = model.to(device)
    
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

if __name__ == "__main__":
    # Initialize model arguments (make sure these match your training configuration)
    model_args = ModelArgs(
        dim=64,
        n_layers=5,
        n_heads=8,
        vocab_size=4096,
        max_seq_len=1024,
    )
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Load the trained model
    model = load_model('transformer_weights.pth', model_args)  # Adjust path as needed
    
    # Example prompts
    prompts = [
        "একটি ছিল রাজা",  # Once upon a time there was a king
        "সুন্দর এক জঙ্গলে",  # In a beautiful forest
        "ছোট্ট মেয়েটি বলল"   # The little girl said
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print("\nPrompt:", prompt)
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=0.8
        )
        print("Generated:", generated)
        print("-" * 50)

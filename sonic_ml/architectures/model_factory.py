from sonic_ml.architectures.llama import Llama

class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(
        architecture: str,
        dim: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        max_seq_len: int,
    )->Llama:
        """
        Create and return a model based on the specified architecture.
        
        Args:
            architecture (str): The name of the model architecture (e.g., "llama")
            dim (int): Model dimension/embedding size
            n_layers (int): Number of transformer layers
            n_heads (int): Number of attention heads
            vocab_size (int): Size of the vocabulary
            max_seq_len (int): Maximum sequence length for input texts
            
        Returns:
            Model architecture instance or None if architecture is not supported
        """
        if architecture.lower() == "llama":
            return Llama(
                dim=dim,
                n_layers=n_layers,
                n_heads=n_heads,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            )
        else:
            raise ValueError(f"Unsupported model architecture: {architecture}")

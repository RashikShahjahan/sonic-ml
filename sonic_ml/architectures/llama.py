import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn.functional as F

class Llama(nn.Module):
    def __init__(self, dim, n_layers, n_heads, vocab_size, max_seq_len):
        super().__init__()
        # Initialize configuration
        self.config = LlamaConfig(
            hidden_size=dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
        )
        
        # Initialize model with causal language modeling head
        self.model = LlamaForCausalLM(self.config)
        self.last_loss = None

    def forward(self, idx, targets=None):
        # Forward pass with labels for loss calculation
        outputs = self.model(idx, labels=targets)
        
        if targets is not None:
            # Training mode
            self.last_loss = outputs.loss
            return None
        else:
            # Inference mode
            return outputs.logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Create optimizer
        # Initialize AdamW optimizer
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None):
        """
        Generate text tokens using the model.
        
        Args:
            idx: Initial token indices (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = more random, close to 0 = more deterministic)
            top_k: If set, only sample from the top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop sequence if too long
            idx_cond = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:]
            
            # Get logits from forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Only use last token's predictions
            
            if temperature == 0.0:
                # Greedy sampling
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Temperature sampling
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append new token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

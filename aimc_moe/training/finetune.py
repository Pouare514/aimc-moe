from __future__ import annotations
import time
from typing import Dict, Any, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aimc_moe.noise.profiles import NoiseProfile
from aimc_moe.training.noisy_forward import wrap_model_noisy, unwrap_model

def aimc_finetune(
    model: nn.Module,
    noise_profile: NoiseProfile,
    dataloader: DataLoader,
    steps: int = 100,
    lr: float = 1e-4,
    loss_fn: Callable[[Any, Any], torch.Tensor] = nn.CrossEntropyLoss(),
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Perform Noise-Aware Finetuning on the model.
    Injects analog noise in the forward pass while training, allowing the optimizer
    to adjust weights to be robust to non-idealities.
    """
    model.to(device)
    
    # 1. Wrap the model in noisy layers
    wrap_model_noisy(model, noise_profile)
    model.train()
    
    # 2. Setup optimizer
    # Standard PyTorch optimizers work directly on the wrapped model's underlying weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    history: List[float] = []
    
    start_time = time.time()
    
    step_count = 0
    pbar = tqdm(total=steps, desc="Noise-Aware Finetuning")
    
    while step_count < steps:
        for x, y in dataloader:
            if step_count >= steps:
                break
                
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (with noise injection)
            # MoE transformer returns (logits, aux_loss), dense transformer returns logits.
            outputs = model(x)
            aux_loss = torch.tensor(0.0, device=device)
            
            if isinstance(outputs, tuple):
                logits, aux_loss = outputs
            else:
                logits = outputs
                
            # Compute loss
            # Handle sequence models output shape (B, S, V) -> loss expects (B*S, V)
            if len(logits.shape) == 3:
                logits = logits.view(-1, logits.shape[-1])
                y = y.view(-1)
                
            loss = loss_fn(logits, y) + 0.1 * aux_loss
            
            # Backward pass & update
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            history.append(loss_val)
            
            pbar.set_postfix({"Loss": f"{loss_val:.4f}"})
            pbar.update(1)
            
            step_count += 1
            
    pbar.close()
    elapsed = time.time() - start_time
    
    # 3. Restore the digital model weights
    unwrap_model(model)
    model.eval()
    
    return {
        "loss_history": history,
        "elapsed_seconds": elapsed,
        "final_loss": history[-1] if history else None
    }

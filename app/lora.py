"""Low-Rank Adaptation (LoRA) layer implementation and adapter injection."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with low-rank adapter matrices A and B."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Initialize LoRA matrices:
        # A: (rank, in_features) with Kaiming uniform
        # B: (out_features, rank) with zeros (starts as identity mapping / zero delta)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward pass (frozen)
        base_out = self.base_layer(x)
        # Low rank delta: (x @ A^T) @ B^T * scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("q_proj", "v_proj", "fc1"),
) -> nn.Module:
    """Inject LoRA adapter layers into targeted linear modules and freeze the rest of the model."""
    # First, freeze all existing parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Recursively find and replace targeted linear modules
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Retrieve parent module and attribute name
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]

                # Replace with LoRALinear wrapper
                lora_layer = LoRALinear(
                    base_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, attr_name, lora_layer)
                break

    return model


def get_trainable_parameters(model: nn.Module) -> dict[str, int | float]:
    """Calculate the total and trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percent = 100.0 * trainable_params / max(all_params, 1)
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": round(percent, 2),
    }


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Save only the trainable LoRA adapter weights."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }
    torch.save(lora_state, target_path)


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Load previously saved LoRA adapter weights."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"LoRA weights file not found: {target_path}")
    lora_state = torch.load(target_path, weights_only=True)
    model.load_state_dict(lora_state, strict=False)

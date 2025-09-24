from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework
        """
        super().__init__(in_features, out_features, bias)
        # LoRA layers: keep in float32 for stability
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        torch.nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.lora_b.weight)
        #torch.nn.init.normal_(self.lora_b.weight, mean=0.0, std=0.01)
        # Make sure only LoRA layers are trainable
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lora_a.parameters():
            p.requires_grad = True
        for p in self.lora_b.parameters():
            p.requires_grad = True
        # Save dtype for main linear
        self.linear_dtype = torch.float16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main output: cast to half, compute, cast back
        main_out = super().forward(x.to(self.linear_dtype))
        # LoRA output: keep in float32
        lora_out = self.lora_b(self.lora_a(x.float()))
        # Sum and cast to input dtype
        out = main_out + lora_out.to(main_out.dtype)
        return out.to(x.dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels,lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels,lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels,lora_dim)
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM,lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM,lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM,lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM,lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM,lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM,lora_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net

from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
        alpha:float=32.0,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        #torch.nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        

        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)        
        # Make sure only LoRA layers are trainable
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lora_a.parameters():
            p.requires_grad = True
        for p in self.lora_b.parameters():
            p.requires_grad = True
        
        self.alpha_div_rank = alpha / lora_dim
        self.linear_dtype = torch.float        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        main_out = super().forward(x.to(self.linear_dtype))
        #main_out = super().forward(x.to(torch.float32)) # self.linear_dtype
        # LoRA output: keep in float32
        #lora_out = self.lora_b(self.lora_a(x.float())) 

        
        if self.lora_a.weight.shape[0] == 0 or self.lora_b.weight.shape[1] == 0:
            lora_out = 0
            out = main_out
        else:
            lora_out = self.alpha_div_rank * self.lora_b(self.lora_a(x.float()))
            out = main_out + lora_out.to(main_out.dtype)
        

        # Sum and cast to input dtype
        #out = main_out + lora_out.to(main_out.dtype)
        return out.to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size, alpha):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size=group_size, alpha=alpha),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size=group_size, alpha=alpha),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size=group_size, alpha=alpha),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16, alpha:float=16.95):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),            
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size=group_size, alpha=alpha),            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net

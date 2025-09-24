import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from pathlib import Path


def block_quantize_3bit(x: torch.Tensor, group_size: int = 8):
    """
    Quantize the input tensor to 3-bit precision along the last dimension.
    Always quantize group_size values together and store their absolute value first.
    x must be 1D and divisible by group_size.
    Returns the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_3 = (x_norm * 7).round().to(torch.uint8)  # 3 bits: 0-7
    # Pack 8 3-bit values into 3 bytes (24 bits)
    packed = torch.zeros(x.size(0), 3, dtype=torch.uint8)
    for i in range(x.size(0)):
        val = 0
        for j in range(group_size):
            val |= (int(x_quant_3[i, j]) & 0x7) << (j * 3)
        packed[i, 0] = (val >> 0) & 0xFF
        packed[i, 1] = (val >> 8) & 0xFF
        packed[i, 2] = (val >> 16) & 0xFF
    return packed, normalization.to(torch.float16)

def block_dequantize_3bit(packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 8):
    """
    Dequantize from 3-bit packed format back to float.
    """
    assert packed.dim() == 2 and packed.size(1) == 3
    normalization = normalization.to(torch.float32)
    x_quant_3 = torch.zeros(packed.size(0), group_size, dtype=torch.float32)
    for i in range(packed.size(0)):
        val = int(packed[i, 0]) | (int(packed[i, 1]) << 8) | (int(packed[i, 2]) << 16)
        for j in range(group_size):
            x_quant_3[i, j] = (val >> (j * 3)) & 0x7
    x_norm = x_quant_3 / 7    
    x = (x_norm * 2 * normalization.to(x_norm.device)) - normalization.to(x_norm.device)
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 8) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # self.register_buffer is used to store the weights in the model, but not as parameters
        # This makes sure weights are put on the correct device when calling `model.to(device)`.
        # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
        # called "weight" that we need to quantize when the model is loaded.
        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, 3, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1,dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            weight = state_dict[f"{prefix}weight"]  # noqa: F841
            del state_dict[f"{prefix}weight"]
            # Quantize the weights and store them in self.weight_q4 and self.weight_norm
            # Flatten the weight matrix to 1D for quantization
            weight_flat = weight.view(-1).to(torch.float32)
            q3, norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(q3)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Dequantize weights
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size)
            # Reshape to (out_features, in_features)
            weight = weight.view(self._shape)
            # Apply linear transformation
            out = torch.nn.functional.linear(x, weight.to(x.device), self.bias)
            return out


class BigNet3Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net





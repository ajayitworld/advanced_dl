import abc

import torch

from .ae import PatchAutoEncoder

#used help from copilot and chatgpt to implement the autoregressive model
def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., embedding_dim) or (..., embedding_dim, h, w)
        returns: (..., codebook_bits) tensor with values in {-1, 1}
        """
        # Handle both channels-last and channels-first formats
        orig_shape = x.shape
        if x.shape[-1] != self.embedding_dim:
            # Convert from channels-first to channels-last
            x = x.permute(0, 2, 3, 1)
        
        # Project and normalize
        x = self.down_proj(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        x = diff_sign(x)
        
        # Restore original channel format if needed
        if orig_shape[-1] != self.embedding_dim:
            x = x.permute(0, 3, 1, 2)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., codebook_bits) tensor with values in {-1, 1}
        returns: (..., embedding_dim) tensor
        """
        # Handle both channels-last and channels-first formats
        orig_shape = x.shape
        if x.shape[-1] != self.codebook_bits:
            # Convert from channels-first to channels-last
            x = x.permute(0, 2, 3, 1)
        
        # Project back to embedding space
        x = self.up_proj(x)
        
        # Restore original channel format if needed
        if orig_shape[-1] != self.codebook_bits:
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self.codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.bsq = BSQ(codebook_bits, latent_dim)
        self.codebook_bits = codebook_bits

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, 3)
        features = super().encode(x)  # (B, h, w, latent_dim)
        if features.shape[-1] != self.bsq.embedding_dim:
            # Convert from channels-first to channels-last if needed
            features = features.permute(0, 2, 3, 1)
        tokens = self.bsq.encode_index(features)  # (B, h, w)
        return tokens

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, h, w) integer tokens
        code = self.bsq._index_to_code(x)  # (B, h, w, codebook_bits)
        features = self.bsq.decode(code)  # (B, h, w, latent_dim)
        img = super().decode(features)  # (B, H, W, 3)
        return img

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, 3)
        features = super().encode(x)  # (B, h, w, latent_dim)
        code = self.bsq.encode(features)  # (B, h, w, codebook_bits)
        return code

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, h, w, codebook_bits) or (B, codebook_bits, h, w)
        if x.shape[-1] != self.codebook_bits:
            # Convert from channels-first to channels-last
            x = x.permute(0, 2, 3, 1)
        features = self.bsq.decode(x)  # (B, h, w, latent_dim)
        img = super().decode(features)  # (B, H, W, 3)
        return img

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        """
        reconstructed = self.decode(self.encode(x))
        tokens = self.encode_index(x)
        cnt = torch.bincount(tokens.flatten(), minlength=2 ** self.codebook_bits)
        stats = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
        }
        return reconstructed, stats

import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)

#used help from copilot and chatgpt to implement the autoregressive model
class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_latent, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, h, w) integer tokens
        B, h, w = x.shape
        seq_len = h * w
        # Flatten to sequence
        x_seq = x.view(B, seq_len)
        # Embed tokens
        x_emb = self.embedding(x_seq)  # (B, seq_len, d_latent)
        # Shift input right by one (prepend zeros)
        pad = torch.zeros((B, 1, self.d_latent), device=x.device)
        x_emb_shifted = torch.cat([pad, x_emb[:, :-1, :]], dim=1)
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        # Transformer
        x_trans = self.transformer(x_emb_shifted, mask=mask)
        # Output logits
        logits = self.output(x_trans)  # (B, seq_len, n_tokens)
        # Reshape back to (B, h, w, n_tokens)
        logits_img = logits.view(B, h, w, self.n_tokens)
        return logits_img, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        # Generate samples autoregressively
        if device is None:
            device = next(self.parameters()).device
        seq_len = h * w
        samples = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        for t in range(seq_len):
            # Embed and shift
            x_emb = self.embedding(samples)
            pad = torch.zeros((B, 1, self.d_latent), device=device)
            x_emb_shifted = torch.cat([pad, x_emb[:, :-1, :]], dim=1)
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            # Transformer
            x_trans = self.transformer(x_emb_shifted, mask=mask)
            logits = self.output(x_trans)  # (B, seq_len, n_tokens)
            # Sample next token
            probs = torch.softmax(logits[:, t, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            samples[:, t] = next_token
        # Reshape to (B, h, w)
        return samples.view(B, h, w)

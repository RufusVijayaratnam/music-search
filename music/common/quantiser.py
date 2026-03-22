from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class QuantiserResult:
    quantised: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class AbstractQuantiser(ABC, torch.nn.Module):
    def __init__(self, codebook_size: int, token_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.codebook_size = codebook_size
        self.token_dim = token_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantiserResult:
        pass


class SimpleQuantiser(AbstractQuantiser):
    def __init__(self, codebook_size: int, token_dim: int, device: torch.device):
        super().__init__(codebook_size, token_dim, device)
        self.codebook = torch.nn.Parameter(
            torch.rand(size=(self.codebook_size, self.token_dim), device=device) - 0.5
        )

    def forward(self, x: torch.Tensor) -> QuantiserResult:
        # x is transformer output, continuous tokens
        # x [batch, token_dim]
        norm = torch.cdist(x, self.codebook, p=2)
        min_idx = torch.argmin(norm, dim=-1)  # non differentiable
        quantised = self.codebook[min_idx]

        # losses,
        commitment: torch.Tensor = torch.norm(
            x - quantised.detach(), dim=-1
        ).mean()  # get the encoder to to target codebook vectors
        codebook: torch.Tensor = torch.norm(
            x.detach() - quantised, dim=-1
        ).mean()  # get the codebook to chase encoder

        # straight through for gradients, pretend encoder output zq directly
        qst: torch.Tensor = x + (quantised - x).detach()
        return QuantiserResult(quantised=qst, codebook_loss=codebook, commitment_loss=commitment)


class ResidualVectorQuantiser(AbstractQuantiser):
    def __init__(
        self, codebook_size: int, token_dim: int, num_codebooks: int, device: torch.device
    ):
        super().__init__(codebook_size, token_dim, device)
        self.num_codebooks = num_codebooks
        self.simple_quants = torch.nn.ModuleList(
            [SimpleQuantiser(codebook_size, token_dim, device) for _ in range(num_codebooks)]
        )

    def forward(self, x: torch.Tensor) -> QuantiserResult:
        cb_loss = 0
        cmt_loss = 0
        residual = x
        quantised_out = torch.zeros_like(x)
        for sq in self.simple_quants:
            res = sq(residual)
            quantised_out = quantised_out + res.quantised
            residual = x - quantised_out.detach()
            cb_loss += res.codebook_loss
            cmt_loss += res.commitment_loss
        return QuantiserResult(
            quantised=quantised_out, codebook_loss=cb_loss, commitment_loss=cmt_loss
        )

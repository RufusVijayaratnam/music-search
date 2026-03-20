import torch
import numpy as np
import torch.nn as nn

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RoPEMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        em: int,
        nheads: int,
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dk = em // nheads
        self.nheads = nheads
        self.em = em
        self.tq = nn.Linear(em, em, bias=bias, **factory_kwargs)
        self.tk = nn.Linear(em, em, bias=bias, **factory_kwargs)
        self.tv = nn.Linear(em, em, bias=bias, **factory_kwargs)
        self.output = nn.Linear(in_features=nheads * self.dk, out_features=em, bias=bias, **factory_kwargs)
        self.rotary = Rotary(self.dk)

    def sdpa(self, q, k, v):
        qkt = torch.einsum("hbij,hbjk->hbik", q, torch.transpose(k, -2, -1))
        sm = torch.softmax(qkt / np.sqrt(self.dk), dim=-1)
        attn = torch.matmul(sm, v)
        return attn

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: [batch, n_tokens, em]
        q, k, v = self.tq(data), self.tk(data), self.tv(data)  # [batch, n_tokens, em]
        q = torch.reshape(q, (*data.shape[:2], self.nheads, self.dk))
        k = torch.reshape(k, (*data.shape[:2], self.nheads, self.dk))
        v = torch.reshape(v, (*data.shape[:2], self.nheads, self.dk))

        cos, sin = self.rotary(q, seq_dim=2)  # seq length is n tokens?

        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        sdpa = self.sdpa(qr, kr, v) # [head, batch, n_tokens, dk]
        sdpa_cat = torch.permute(sdpa, dims=(1, 2, 0, 3)).flatten(start_dim=-2, end_dim=-1)
        return self.output(sdpa_cat)


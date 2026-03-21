import torch
import torch.nn as nn
import torch.nn.functional as F


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
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


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
        self.output = nn.Linear(
            in_features=nheads * self.dk, out_features=em, bias=bias, **factory_kwargs
        )
        self.rotary = Rotary(self.dk)

    def sdpa(self, q, k, v):
        # q, k, v: [batch, nheads, seq, dk]
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: [batch, seq, em]
        b, s, _ = data.shape
        q, k, v = self.tq(data), self.tk(data), self.tv(data)  # [batch, seq, em]
        q = q.reshape(b, s, self.nheads, self.dk).transpose(1, 2)  # [batch, nheads, seq, dk]
        k = k.reshape(b, s, self.nheads, self.dk).transpose(1, 2)
        v = v.reshape(b, s, self.nheads, self.dk).transpose(1, 2)

        cos, sin = self.rotary(q, seq_dim=2)  # cos/sin: [1, 1, seq, dk]

        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        sdpa = self.sdpa(qr, kr, v)  # [batch, nheads, seq, dk]
        sdpa_t = sdpa.transpose(1, 2).reshape(b, s, self.nheads * self.dk)  # [batch, seq, em]
        return self.output(sdpa_t)

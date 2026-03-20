import torch
from music.common.loader import AudioData


def gen_batch(data: AudioData, E_q, E_k, E_v, device, dtype=torch.float32, query_seq_len_1=False):
    # generate semi-realistic data using Zipf distribution for sentence lengths
    sentence_lengths = data.lengths

    # Note: the torch.jagged layout is a nested tensor layout that supports a single ragged
    # dimension and works with torch.compile. The batch items each have shape (B, S*, D)
    # where B = batch size, S* = ragged sequence length, and D = embedding dimension.
    if query_seq_len_1:
        query = torch.nested.nested_tensor(
            [torch.randn(1, E_q, dtype=dtype, device=device) for l in sentence_lengths],
            layout=torch.jagged,
        )
    else:
        query = torch.nested.nested_tensor(
            [torch.randn(int(l.item()), E_q, dtype=dtype, device=device) for l in sentence_lengths],
            layout=torch.jagged,
        )

    key = torch.nested.nested_tensor(
        [torch.randn((s.item()), E_k, dtype=dtype, device=device) for s in sentence_lengths],
        layout=torch.jagged,
    )

    value = torch.nested.nested_tensor(
        [torch.randn(int(s.item()), E_v, dtype=dtype, device=device) for s in sentence_lengths],
        layout=torch.jagged,
    )

    return query, key, value, sentence_lengths

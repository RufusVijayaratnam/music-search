from music.common.loader import load_audio_data, AudioData
from music.common.quantiser import SimpleQuantiser
import torch

device = torch.device("cuda")
dtype = torch.float32

# data = load_audio_data("./data/train/", device=device, dtype=dtype)

token_dim = 8
q = SimpleQuantiser(20, token_dim, device)

tokens = torch.rand(size=(10, token_dim), device=device)

q(tokens)

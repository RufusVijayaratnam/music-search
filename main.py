from music.common.loader import load_audio_data, AudioData
import torch

device = torch.device("mps")
dtype = torch.float32

data = load_audio_data("./data/train/", device=device, dtype=dtype)
breakpoint()

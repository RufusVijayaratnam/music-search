import torch
import os
import numpy as np
import stempeg


class AudioData:
    obs: torch.Tensor

    def __init__(self, obs: torch.Tensor, voice_mask: torch.Tensor):
        self.voice_mask = voice_mask
        self.obs = obs

    def __get_item__(self, i: int) -> AudioData:
        return AudioData(self.voice_mask[i], self.obs[i])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        full_obs = self.obs.to(device=device, dtype=dtype)
        voice_obs = self.voice_mask.to(device=device, dtype=dtype)
        return AudioData(voice_obs, full_obs)


def load_audio_data(path: str, device: torch.device, dtype: torch.dtype) -> AudioData:
    max_len = 1_000_000
    files = [os.path.join(path, fp) for fp in os.listdir(path) if fp.endswith("mp4")]
    obs_all = torch.empty(size=(len(files), max_len), dtype=dtype, device=device)
    voice_mask_all = torch.empty(size=(len(files), max_len), dtype=dtype, device=device)
    for i, fp in enumerate(files):
        stems, rate = stempeg.read_stems(fp)
        mixture_mono = torch.from_numpy(stems[0].mean(axis=-1))[:max_len]
        voice_mono = stems[4].mean(axis=-1)[:max_len]
        length = len(voice_mono)
        window = 4410
        threshold = 0.05
        rms = torch.from_numpy(
            np.sqrt(np.convolve(voice_mono**2, np.ones(window) / window, mode="same"))
        )
        voice_mask = rms > threshold
        obs_all[i, : min(max_len, length)] = mixture_mono
        voice_mask
    return AudioData(obs, voice_mask).to(device=device, dtype=dtype)
    pass

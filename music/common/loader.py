from numpy.typing import NDArray
import torch
import os
import numpy as np
import stempeg
from scipy.ndimage import uniform_filter1d
from multiprocessing.pool import ThreadPool
import tqdm
from dataclasses import dataclass


class AudioData:
    mixtures: torch.Tensor
    voice_mask: torch.Tensor
    lengths: torch.Tensor

    def __init__(self, mixtures: torch.Tensor, voice_mask: torch.Tensor, lengths: torch.Tensor):
        if voice_mask.device != mixtures.device:
            raise ValueError(
                f"voice_mask device ({voice_mask.device}) must be the same as mixtures device ({mixtures.device})"
            )
        if lengths.device != mixtures.device:
            raise ValueError(
                f"voice_mask device ({voice_mask.device}) must be the same as mixtures device ({mixtures.device})"
            )
        self.voice_mask = voice_mask
        self.mixtures = mixtures
        self.lengths = lengths

    def __getitem__(self, i) -> "AudioData":
        if isinstance(i, int):
            i = slice(i, i + 1)
        return AudioData(self.mixtures[i], self.voice_mask[i], self.lengths[i])

    def __len__(self) -> int:
        return len(self.mixtures)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        mixtures = self.mixtures.to(device=device, dtype=dtype)
        voice_mask = self.voice_mask.to(device=device, dtype=dtype)
        lengths = self.lengths.to(device=device)
        return AudioData(mixtures, voice_mask, lengths)


def load_audio_data(path: str, device: torch.device, dtype: torch.dtype) -> AudioData:
    max_len = 1_000_000
    files = [os.path.join(path, fp) for fp in os.listdir(path) if fp.endswith("mp4")]

    @dataclass
    class _AudiData:
        mixture: NDArray
        voice_mask: NDArray
        length: int

    def _load_audio_data(filepath: str):
        info = stempeg.Info(filepath)
        mixture_rate = info.rate(0)
        voice_rate = info.rate(4)
        if mixture_rate != voice_rate:
            raise ValueError("mixture and voice rate not the same")

        duration = max_len / mixture_rate
        stems, _ = stempeg.read_stems(
            filepath, duration=duration, stem_id=[0, 4], info=info, dtype=np.float32
        )
        mixture_padded = np.zeros(shape=(2, max_len), dtype=np.float32)
        voice_mask_padded = np.zeros(shape=(max_len,), dtype=np.float32)
        mixture_stereo = stems[0][:max_len].T  # [2, length]
        voice_mono = stems[1][:max_len].mean(axis=-1)
        length = mixture_stereo.shape[1]

        window = 4410
        threshold = 0.05
        rms = np.sqrt(np.maximum(uniform_filter1d(voice_mono**2, size=window), 0.0))
        voice_mask = rms > threshold

        mixture_padded[:, :length] = mixture_stereo
        voice_mask_padded[:length] = voice_mask

        return _AudiData(mixture=mixture_padded, voice_mask=voice_mask_padded, length=length)


    _load_audio_data(files[0])
    with ThreadPool() as p:
        audio_obs_inputs = list(
            tqdm.tqdm(
                p.imap(_load_audio_data, files),
                total=len(files),
                desc="Loading audio",
            )
        )

    mixtures = [ad.mixture for ad in audio_obs_inputs]
    voice_masks = [ad.voice_mask for ad in audio_obs_inputs]
    lengths = torch.tensor([ad.length for ad in audio_obs_inputs], device=device)

    mixtures_all = torch.from_numpy(np.stack(mixtures)).to(device, dtype)
    voice_masks_all = torch.from_numpy(np.stack(voice_masks)).to(device, dtype)
    return AudioData(mixtures_all, voice_masks_all, lengths)

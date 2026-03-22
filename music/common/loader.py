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
    device: torch.device

    def __init__(
        self,
        mixtures: torch.Tensor,
        voice_mask: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ):
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
        self.device = device

    def __getitem__(self, i) -> "AudioData":
        if isinstance(i, int):
            i = slice(i, i + 1)
        return AudioData(self.mixtures[i], self.voice_mask[i], self.lengths[i], self.device)

    def __len__(self) -> int:
        return len(self.mixtures)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        mixtures = self.mixtures.to(device=device, dtype=dtype)
        voice_mask = self.voice_mask.to(device=device, dtype=dtype)
        lengths = self.lengths.to(device=device)
        return AudioData(mixtures, voice_mask, lengths, device or self.device)

    def sample_batch(self, batch_size: int, window_len: int) -> "AudioData":
        num_tracks = self.mixtures.shape[0]
        min_len = int(torch.min(self.lengths).item())
        if min_len < window_len:
            raise ValueError(
                f"Can't have window len longer than a track, min track len is {min_len}"
            )

        track_indices = torch.randint(0, num_tracks, size=(batch_size,), device=self.device)
        track_lengths = self.lengths[track_indices]
        max_starts = torch.clamp(track_lengths - window_len, min=0)
        rand_vals = torch.rand(batch_size, device=self.device)
        start_indices = (rand_vals * max_starts.float()).long()

        offsets = self.mixtures.offsets()
        flat_starts = offsets[track_indices] + start_indices

        flat_indices = flat_starts.unsqueeze(1) + torch.arange(window_len, device=self.device)
        flat_indices = flat_indices.reshape(-1)

        _, channels, _ = self.mixtures.shape
        mixtures_values = self.mixtures.values()  # [channels, total_samples]
        mixtures_indexed = mixtures_values[:, flat_indices]  # [channels, batch_size * window_len]
        mixtures_flat = mixtures_indexed.reshape(channels, batch_size, window_len).permute(
            1, 0, 2
        )  # [batch_size, channels, window_len]
        voice_masks_flat = self.voice_mask.values()[flat_indices].reshape(batch_size, window_len)

        return AudioData(mixtures_flat, voice_masks_flat, track_lengths, self.device)


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
        mixture_stereo = stems[0][:max_len].T  # [2, length]
        voice_mono = stems[1][:max_len].mean(axis=-1)
        length = mixture_stereo.shape[1]

        window = 4410
        threshold = 0.05
        rms = np.sqrt(np.maximum(uniform_filter1d(voice_mono**2, size=window), 0.0))
        voice_mask = rms > threshold

        return _AudiData(mixture=mixture_stereo, voice_mask=voice_mask, length=length)

    with ThreadPool() as p:
        audio_obs_inputs = list(
            tqdm.tqdm(
                p.imap(_load_audio_data, files),
                total=len(files),
                desc="Loading audio",
            )
        )

    mixtures = torch.nested.nested_tensor(
        [ad.mixture for ad in audio_obs_inputs], layout=torch.jagged
    )
    voice_masks = torch.nested.nested_tensor(
        [ad.voice_mask for ad in audio_obs_inputs], layout=torch.jagged
    )
    lengths = torch.tensor([ad.length for ad in audio_obs_inputs], device=device)

    return AudioData(mixtures, voice_masks, lengths, device=device)

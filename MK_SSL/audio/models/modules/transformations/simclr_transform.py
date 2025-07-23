import os
import random
import torch
import torchaudio
import torchaudio.transforms as AT
from pathlib import Path
from torchaudio.functional import resample
from torchaudio.sox_effects import apply_effects_tensor
from typing import Tuple


class MUSANNoiseAdder:
    def __init__(self, musan_root: str, snr_range_db=(5, 10), sample_rate=16000):
        self.sample_rate = sample_rate
        self.snr_range_db = snr_range_db
        self.noise_paths = list(Path(musan_root).rglob("*.wav"))
        if not self.noise_paths:
            raise FileNotFoundError(f"No .wav files found under {musan_root}")

    def _load_random_noise(self, target_len: int) -> torch.Tensor:
        path = random.choice(self.noise_paths)
        waveform, orig_sr = torchaudio.load(str(path))
        if orig_sr != self.sample_rate:
            waveform = resample(waveform, orig_sr, self.sample_rate)
        if waveform.size(1) < target_len:
            waveform = waveform.repeat(1, (target_len // waveform.size(1)) + 1)
        return waveform[:, :target_len]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to a waveform or a batch of waveforms.
        Args:
            waveform: [1, L] or [B, 1, L]
        """
        if waveform.dim() == 2:  # [1, L]
            return self._add_noise_single(waveform)
        elif waveform.dim() == 3:  # [B, 1, L]
            return torch.stack([self._add_noise_single(w) for w in waveform])
        else:
            raise ValueError(f"Expected [1, L] or [B, 1, L], but got {tuple(waveform.shape)}.")

    def _add_noise_single(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = self._load_random_noise(waveform.size(1))
        snr_db = random.uniform(*self.snr_range_db)
        rms_signal = waveform.pow(2).mean().sqrt()
        rms_noise = noise.pow(2).mean().sqrt()
        scale = rms_signal / (10 ** (snr_db / 20)) / (rms_noise + 1e-6)
        return waveform + scale * noise


class SimCLRAudioTransform:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        time_mask_param=40,
        freq_mask_param=10,
        use_musan=False,
        musan_root="musan/noise",
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        if use_musan:
            self._download_and_extract_musan(musan_root)
            self.noise_adder = MUSANNoiseAdder(musan_root, sample_rate=sample_rate)
        else:
            self.noise_adder = self._add_gaussian_noise

    def _download_and_extract_musan(self, root: str):
        url = "http://www.openslr.org/resources/17/musan.tar.gz"
        archive_path = "musan.tar.gz"
        extract_path = Path(root).parent

        if not Path(root).exists():
            try:
                import urllib.request
                import tarfile
                urllib.request.urlretrieve(url, archive_path)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=extract_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download or extract MUSAN dataset: {e}")
            finally:
                if Path(archive_path).exists():
                    os.remove(archive_path)

    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:  # [1, L]
            return self._add_gaussian_noise_single(waveform)
        elif waveform.dim() == 3:  # [B, 1, L]
            return torch.stack([self._add_gaussian_noise_single(w) for w in waveform])
        else:
            raise ValueError(f"Expected [1, L] or [B, 1, L], but got {tuple(waveform.shape)}.")

    def _add_gaussian_noise_single(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform)
        snr = random.uniform(5, 10)
        rms_signal = waveform.pow(2).mean().sqrt()
        rms_noise = rms_signal / (10 ** (snr / 20))
        return waveform + rms_noise * noise

    def _speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        # This must be applied per sample (not batch).
        if waveform.dim() == 3:
            return torch.stack([self._speed_perturb_single(w) for w in waveform])
        return self._speed_perturb_single(waveform)

    def _speed_perturb_single(self, waveform: torch.Tensor) -> torch.Tensor:
        speed = random.uniform(0.8, 1.2)
        new_sr = int(self.sample_rate * speed)
        waveform = resample(waveform, self.sample_rate, new_sr)
        return resample(waveform, new_sr, self.sample_rate)

    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            return torch.stack([self._pitch_shift_single(w) for w in waveform])
        return self._pitch_shift_single(waveform)

    def _pitch_shift_single(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-300, 300)  # in cents
        try:
            return torchaudio.functional.pitch_shift(waveform, self.sample_rate, shift)
        except Exception:
            return waveform

    def _room_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            return torch.stack([self._room_reverb_single(w) for w in waveform])
        return self._room_reverb_single(waveform)

    def _room_reverb_single(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        room_scale = random.randint(0, 100)
        effects = [["reverb", "50", "50", str(room_scale)]]
        try:
            augmented_waveform, _ = apply_effects_tensor(waveform, self.sample_rate, effects)
            return augmented_waveform
        except Exception:
            return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            waveform = self.noise_adder(waveform)
        if random.random() < 0.5:
            waveform = self._pitch_shift(waveform)
        if random.random() < 0.5:
            waveform = self._speed_perturb(waveform)
        if random.random() < 0.5:
            waveform = self._room_reverb(waveform)
        return waveform

    def __call__(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: [1, L] or [B, 1, L]
        Returns:
            Two augmented views of the same input with shape [1, L] or [B, 1, L].
        """
        if waveform.dim() not in (2, 3):
            raise ValueError(f"Expected [1, L] or [B, 1, L], but got {tuple(waveform.shape)}.")

        x1 = self._augment_waveform(waveform.clone())
        x2 = self._augment_waveform(waveform.clone())
        return x1, x2

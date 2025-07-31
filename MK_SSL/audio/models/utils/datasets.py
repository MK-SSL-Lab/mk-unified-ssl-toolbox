import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
from typing import Dict
import logging
from torch import nn


class HuBERTWrapperDataset(Dataset):
    """A dataset wrapper for HuBERT pre-training.

    This class wraps an existing dataset to enable dynamic pseudo-label updates 
    and ensures that pseudo-labels are aligned with the feature extractor's output 
    length. It also performs validation on dataset structure.

    Attributes:
        original_dataset (Dataset): The user's original dataset.
        feature_extractor (nn.Module): HuBERT ConvFeatureExtractor instance.
        sample_rate (int): Audio sample rate (default is 16kHz).
        pseudo_labels_dict (Dict[int, np.ndarray]): Dictionary of pseudo-labels indexed by dataset index.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, original_dataset: Dataset, feature_extractor, sample_rate: int = 16000, logger=None):
        self.original_dataset = original_dataset
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.pseudo_labels_dict: Dict[int, np.ndarray] = {}
        self.logger = logger if logger is not None else self._get_default_logger()

        # Track number of samples for safety
        self.dataset_length = len(self.original_dataset)
        self.logger.info(f"HuBERTWrapperDataset initialized with {self.dataset_length} samples.")

    def _get_default_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

    def __len__(self) -> int:
        return self.dataset_length

    def set_pseudo_labels(self, pseudo_labels_dict: Dict[int, np.ndarray]):
        """Updates the internal pseudo-label dictionary."""
        self.pseudo_labels_dict = pseudo_labels_dict
        self.logger.info(f"Pseudo-labels updated successfully for {len(pseudo_labels_dict)} samples.")

    def _align_pseudo_labels(self, pseudo_label: np.ndarray, audio_len: int) -> np.ndarray:
        """Aligns pseudo-labels to match feature extractor output length T'."""
        target_len = self.feature_extractor.get_output_lengths(
            torch.tensor([audio_len])
        ).item()

        if len(pseudo_label) < target_len:
            pad_len = target_len - len(pseudo_label)
            pseudo_label = np.pad(pseudo_label, (0, pad_len), mode='edge')
        elif len(pseudo_label) > target_len:
            pseudo_label = pseudo_label[:target_len]

        return pseudo_label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve a sample with a guaranteed unique original index."""
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(f"Index {idx} is out of bounds for dataset length {self.dataset_length}.")

        original_item = self.original_dataset[idx]
        if original_item is None:
            raise RuntimeError(f"Dataset returned None for index {idx}.")

        if not isinstance(original_item, dict) or "audio" not in original_item or "length" not in original_item:
            raise KeyError(f"Dataset must return a dict with ['audio', 'length'], got {type(original_item)}.")

        audio_tensor = original_item["audio"]
        length = original_item["length"]

        if not torch.is_tensor(audio_tensor):
            raise TypeError(f"Expected 'audio' to be a torch.Tensor but got {type(audio_tensor)}.")

        # if audio_tensor.ndim > 1:
        #     self.logger.warning(f"Audio tensor for index {idx} has {audio_tensor.ndim} dims. Squeezing.")
        #     audio_tensor = audio_tensor.squeeze()

        # if audio_tensor.ndim != 1:
        #     raise ValueError(f"Audio tensor for index {idx} is not 1D after squeezing: {audio_tensor.shape}.")

        sample_dict = {"audio": audio_tensor, "length": length, "original_idx": idx}
        
        padded_len = audio_tensor.shape[1]

        if self.pseudo_labels_dict:
            pseudo_label = self.pseudo_labels_dict.get(idx)
            if pseudo_label is None:
                self.logger.error(f"Pseudo-labels for index {idx} not found.")
                raise KeyError(f"Pseudo-labels for index {idx} not found.")

            pseudo_label = self._align_pseudo_labels(pseudo_label, audio_len=padded_len)
            sample_dict["pseudo_labels"] = torch.from_numpy(pseudo_label).long()

        return sample_dict

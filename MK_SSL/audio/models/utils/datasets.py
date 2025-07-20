import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import logging


class HuBERTWrapperDataset(Dataset):
    def __init__(self, original_dataset: Dataset, target_frame_rate: int = 50, sample_rate: int = 16000, logger=None):
        """
        A wrapper dataset for HuBERT pre-training that handles dynamic pseudo-label updates
        and ensures pseudo-labels match feature extractor output length.

        Args:
            original_dataset (Dataset): The user's original dataset.
            target_frame_rate (int): Frame rate after the ConvFeatureExtractor (e.g., 50Hz for 20ms frames).
            sample_rate (int): Audio sample rate (default 16kHz).
            logger (logging.Logger, optional): Logger instance.
        """
        self.original_dataset = original_dataset
        self.target_frame_rate = target_frame_rate
        self.sample_rate = sample_rate
        self.pseudo_labels_dict: Dict[int, np.ndarray] = {}
        self.logger = logger if logger is not None else self._get_default_logger()

    def _get_default_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

    def __len__(self) -> int:
        return len(self.original_dataset)

    def set_pseudo_labels(self, pseudo_labels_dict: Dict[int, np.ndarray]):
        """
        Updates the internal pseudo-labels dictionary, mapping original_idx to label sequences.
        """
        self.pseudo_labels_dict = pseudo_labels_dict
        self.logger.info("HuBERTWrapperDataset: Pseudo-labels updated successfully.")

    def _align_pseudo_labels(self, pseudo_label: np.ndarray, audio_len: int) -> np.ndarray:
        """
        Align pseudo-label sequence length to match feature length T' produced by ConvFeatureExtractor.

        Args:
            pseudo_label (np.ndarray): Original pseudo-label sequence.
            audio_len (int): Original audio length (samples).

        Returns:
            np.ndarray: Adjusted pseudo-label sequence of length T'.
        """
        target_len = int((audio_len / self.sample_rate) * self.target_frame_rate)

        if len(pseudo_label) < target_len:
            pad_len = target_len - len(pseudo_label)
            pseudo_label = np.pad(pseudo_label, (0, pad_len), mode='edge')
        elif len(pseudo_label) > target_len:
            pseudo_label = pseudo_label[:target_len]

        return pseudo_label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_item = self.original_dataset[idx]

        if original_item is None:
            raise RuntimeError(f"Dataset returned None for index {idx}.")

        # Extract audio and length from original dataset
        if isinstance(original_item, dict):
            if "audio" not in original_item or "length" not in original_item:
                raise KeyError(f"Original dataset must return keys ['audio', 'length']. Got: {list(original_item.keys())}")
            audio_tensor = original_item["audio"]
            length = original_item["length"]
        else:
            raise TypeError("Original dataset must return a dictionary with keys ['audio', 'length'].")

        if not torch.is_tensor(audio_tensor):
            raise TypeError(f"Expected 'audio' to be a torch.Tensor but got {type(audio_tensor)}.")

        if audio_tensor.ndim > 1:
            self.logger.warning(f"Audio tensor for index {idx} has {audio_tensor.ndim} dimensions. Squeezing.")
            audio_tensor = audio_tensor.squeeze()

        if audio_tensor.ndim != 1:
            raise ValueError(f"Audio tensor for index {idx} is not 1D after squeezing: {audio_tensor.shape}.")

        # Return audio, length, and pseudo_labels (if available)
        if self.pseudo_labels_dict:
            pseudo_label = self.pseudo_labels_dict.get(idx)
            if pseudo_label is None:
                self.logger.error(f"Pseudo-labels for index {idx} not found.")
                raise KeyError(f"Pseudo-labels for index {idx} not found.")

            pseudo_label = self._align_pseudo_labels(pseudo_label, audio_len=length)
            pseudo_label_tensor = torch.from_numpy(pseudo_label).long()
            return {
                "audio": audio_tensor,
                "length": length,
                "pseudo_labels": pseudo_label_tensor
            }
        else:
            return {
                "audio": audio_tensor,
                "length": length,
                "original_idx": idx
            }

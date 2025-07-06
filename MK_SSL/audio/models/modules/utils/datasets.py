# File: MK_SSL/audio/datasets/hubert_wrapper_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import logging

class HuBERTWrapperDataset(Dataset):
    def __init__(self, original_dataset: Dataset, logger=None): # Removed audio_paths from constructor
        """
        A wrapper dataset for HuBERT pre-training that handles dynamic pseudo-label updates
        without assuming a specific __getitem__ return structure for the original dataset.

        Args:
            original_dataset (Dataset): The user's original dataset.
                                        Its __getitem__ should return an audio tensor,
                                        or a structure (dict/tuple) from which an audio tensor can be extracted.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
        """
        self.original_dataset = original_dataset
        self.pseudo_labels_dict: Dict[int, np.ndarray] = {} # Maps original_idx to pseudo_labels

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_item = self.original_dataset[idx]
        audio_tensor = None

        # Robustly try to extract the audio tensor from various return types
        if torch.is_tensor(original_item):
            audio_tensor = original_item
        elif isinstance(original_item, dict):
            # Try common keys for audio (case-insensitive)
            for key in ['audio', 'wav', 'input', 'waveform']:
                if key in original_item and torch.is_tensor(original_item[key]):
                    audio_tensor = original_item[key]
                    break
                elif key.capitalize() in original_item and torch.is_tensor(original_item[key.capitalize()]):
                    audio_tensor = original_item[key.capitalize()]
                    break
            if audio_tensor is None:
                # Fallback: if only one tensor in dict, assume it's audio
                tensor_values = [v for v in original_item.values() if torch.is_tensor(v)]
                if len(tensor_values) == 1:
                    audio_tensor = tensor_values[0]
                else:
                    raise TypeError(f"Could not identify a clear audio tensor in dictionary item for index {idx}. "
                                    "Please ensure your dataset returns a tensor directly, or in a dictionary under "
                                    "a common key (e.g., 'audio', 'wav'), or as the only tensor in the dictionary.")
        elif isinstance(original_item, (tuple, list)):
            # Try the first element, then search for any tensor
            for element in original_item:
                if torch.is_tensor(element):
                    audio_tensor = element
                    break
            if audio_tensor is None:
                raise TypeError(f"Could not identify an audio tensor in tuple/list item for index {idx}. "
                                "Please ensure your dataset returns a tensor directly, or as one of the elements "
                                "in the tuple/list (preferably the first).")
        else:
            raise TypeError(f"Unsupported __getitem__ return type for index {idx}: {type(original_item)}. "
                            "Expected a Tensor, dict containing a Tensor, or tuple/list containing a Tensor.")

        if audio_tensor is None: # Should not happen if previous checks are robust
            raise RuntimeError(f"Failed to extract audio tensor for index {idx}. Please check your dataset's __getitem__ output.")

        # Ensure audio_tensor is 1D (sequence) as expected by feature extractors (batch dim added by DataLoader's collate_fn)
        if audio_tensor.ndim > 1:
            self.logger.warning(f"Audio tensor for index {idx} has {audio_tensor.ndim} dimensions. Squeezing to 1D. "
                                "Ensure your dataset returns audio as a 1D sequence (e.g., [T]).")
            audio_tensor = audio_tensor.squeeze()
        if audio_tensor.ndim != 1:
             raise ValueError(f"Extracted audio tensor for index {idx} is not 1D after squeezing: {audio_tensor.shape}. Expected (T,).")


        # If pseudo_labels_dict is populated, return audio and pseudo_labels.
        # Otherwise, return audio and original_idx for the initial K-means pass.
        if self.pseudo_labels_dict:
            pseudo_label = self.pseudo_labels_dict.get(idx)
            if pseudo_label is None:
                self.logger.error(f"Pseudo-labels for index {idx} not found in the dictionary. This should not happen if set_pseudo_labels was called correctly.")
                raise KeyError(f"Pseudo-labels for index {idx} not found.")
            pseudo_label_tensor = torch.from_numpy(pseudo_label).long()
            return {"audio": audio_tensor, "pseudo_labels": pseudo_label_tensor}
        else:
            return {"audio": audio_tensor, "original_idx": idx}
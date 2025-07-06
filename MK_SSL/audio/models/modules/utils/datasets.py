import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import logging

class HuBERTWrapperDataset(Dataset):
    def __init__(self, original_dataset: Dataset, audio_paths: List[str], logger=None):
        """
        A wrapper dataset for HuBERT pre-training that handles dynamic pseudo-label updates.

        Args:
            original_dataset (Dataset): The user's original dataset.
                                        Its __getitem__ must return (audio_tensor, audio_file_path_string).
            audio_paths (List[str]): A list of all audio file paths in the dataset,
                                     matching the order of the original_dataset's indices.
                                     This is crucial for mapping pseudo-labels.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
        """
        self.original_dataset = original_dataset
        self.audio_paths = audio_paths
        self.pseudo_labels_dict: Dict[str, np.ndarray] = {}

        self.logger = logger if logger is not None else self._get_default_logger()

        if len(self.original_dataset) != len(self.audio_paths):
            raise ValueError(
                "The length of the original_dataset must match the length of audio_paths provided. "
                "Ensure audio_paths are ordered consistently with dataset indices."
            )

    def _get_default_logger(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

    def __len__(self) -> int:
        return len(self.original_dataset)

    def set_pseudo_labels(self, pseudo_labels_dict: Dict[str, np.ndarray]):
        """
        Updates the internal pseudo-labels dictionary.
        """
        self.pseudo_labels_dict = pseudo_labels_dict
        self.logger.info("HuBERTWrapperDataset: Pseudo-labels updated successfully.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Expect user's original dataset __getitem__ to return (audio_tensor, audio_file_path)
        item_data = self.original_dataset[idx]

        if not (isinstance(item_data, tuple) and len(item_data) == 2):
            raise TypeError(
                "For HuBERT training with HuBERTWrapperDataset, your original dataset's __getitem__ "
                "must return a tuple of (audio_tensor, audio_file_path_string) to map pseudo-labels. "
                f"Received: {type(item_data)}"
            )

        audio, audio_path = item_data

        if not isinstance(audio_path, str):
            raise TypeError(
                "For HuBERT training with HuBERTWrapperDataset, the second item returned by your "
                "original dataset's __getitem__ must be the audio_file_path as a string. "
                f"Received type for audio_path: {type(audio_path)}"
            )

        # Retrieve the pseudo-label for the current audio_path
        if not self.pseudo_labels_dict:
            raise RuntimeError(
                "Pseudo-labels have not been set for the HuBERTWrapperDataset. "
                "Ensure `set_pseudo_labels` is called before iterating over the DataLoader."
            )
        
        try:
            pseudo_label = self.pseudo_labels_dict[audio_path]
        except KeyError:
            self.logger.error(f"Pseudo-labels not found for audio path: {audio_path}. This path should be present in `audio_paths_for_kmeans`.")
            raise KeyError(f"Pseudo-labels for path {audio_path} not found in the dictionary. Ensure all paths provided to `audio_paths_for_kmeans` are correctly handled by your dataset and pseudo-label generation.")

        # Convert numpy array pseudo_label to torch.LongTensor as expected by HuBERTLoss
        pseudo_label_tensor = torch.from_numpy(pseudo_label).long()

        return {"audio": audio, "pseudo_labels": pseudo_label_tensor}
import os
import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm # Using tqdm.auto for consistency
from typing import Dict, List, Literal, Optional
import logging # For adding logger

class PseudoLabelGenerator:
    """
    Generates HuBERT pseudo-labels (hidden units) from MFCCs or Transformer layer outputs.
    This refactored version handles K-means fitting and label generation in a unified way,
    and returns labels directly for dynamic dataset updates.
    """

    def __init__(
        self,
        kmeans_clusters: int = 100,
        sample_rate: int = 16000,
        save_dir: str = "generated_labels",
        logger=None # Added logger argument
    ):
        self.kmeans_clusters = kmeans_clusters
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        self.logger = logger if logger is not None else self._get_default_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        # MiniBatchKMeans is used; for extremely large datasets, iterative partial_fit
        # with a DataLoader would be more memory efficient than collecting all_features first.
        # For now, we collect all features for the initial fit as in the original design.
        self.kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=1024, random_state=0, n_init='auto')
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        self.fitted = False # Indicates if K-means has been fitted for the current context

        # Internal state for current feature extraction mode
        self._current_input_type: Optional[Literal["mfcc", "transformer"]] = None
        self._current_model: Optional[torch.nn.Module] = None
        self._current_transformer_layer: Optional[int] = None

    def _get_default_logger(self):
        """Creates a basic logger if none is provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

    def set_current_iteration_mode(
        self,
        is_mfcc: bool,
        model: Optional[torch.nn.Module] = None,
        transformer_layer: Optional[int] = None,
    ):
        """
        Configures the feature extraction mode for the current iteration (MFCC or Transformer-based).
        """
        if is_mfcc:
            self._current_input_type = "mfcc"
            self._current_model = None
            self._current_transformer_layer = None
        else:
            if model is None:
                raise ValueError("Model must be provided for transformer-based feature extraction.")
            self._current_input_type = "transformer"
            self._current_model = model # Model should already be in eval mode from Trainer.
            self._current_transformer_layer = transformer_layer # Can be None, model should handle it

    def _extract_features_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Helper to extract features based on the currently set mode.
        Expects a single audio waveform (e.g., (1, N_samples)) or (N_samples,).
        Returns features with a batch dimension (B=1, T, D).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # Add batch dimension if missing

        if self._current_input_type == "mfcc":
            # MFCC extractor expects (B, N_samples)
            mfcc = self.mfcc_extractor(audio).transpose(1, 2)  # (B, T, 13)
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)
            return torch.cat([mfcc, delta, delta2], dim=-1) # (B, T, 39)

        elif self._current_input_type == "transformer":
            if self._current_model is None:
                raise RuntimeError("Model not set for transformer-based feature extraction.")
            with torch.no_grad():
                # self._current_model.feature_extractor expects (B, N_samples)
                feat = self._current_model.feature_extractor(audio)  # (B, C, T')
                feat = feat.transpose(1, 2)  # (B, T', C)
                # self._current_model.encoder.extract_layer
                enc = self._current_model.encoder.extract_layer(feat, self._current_transformer_layer)  # (B, T', D)
            return enc # (B, T', D)
        else:
            raise ValueError(f"Feature extraction mode not set or unsupported: {self._current_input_type}")

    def generate_pseudo_labels(
        self,
        audio_paths: List[str],
        model: Optional[torch.nn.Module] = None,
        is_mfcc: bool = True,
        transformer_layer: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, np.ndarray]:
        """
        Generates pseudo-labels for a list of audio paths, performing K-means fitting if necessary.
        Returns a dictionary mapping audio paths to their generated pseudo-label sequences.
        Labels are also saved to disk as individual .npy files.

        Args:
            audio_paths (List[str]): List of paths to audio files for pseudo-label generation.
            model (Optional[torch.nn.Module]): The HuBERT model to use for feature extraction
                                                in transformer-based iterations.
            is_mfcc (bool): If True, uses MFCCs; otherwise, uses model's hidden states.
            transformer_layer (Optional[int]): The specific transformer layer to extract features from
                                               if `is_mfcc` is False.
            device (torch.device): The device to perform feature extraction on.
        """
        self.set_current_iteration_mode(is_mfcc, model, transformer_layer)
        
        all_features_for_kmeans = []
        self.logger.info("Collecting features for K-means fitting...")
        for path in tqdm(audio_paths, desc="Extracting features"):
            wav, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                wav = torchextra.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate) # Use torchextra.functional.resample if available, otherwise torchaudio.transforms.Resample
            wav = wav.to(device)
            
            extracted_feature = self._extract_features_from_audio(wav) # (B=1, T, D)
            all_features_for_kmeans.append(extracted_feature.squeeze(0).cpu().numpy()) # Store (T, D)

        flat_features = np.concatenate(all_features_for_kmeans, axis=0)
        
        self.logger.info(f"Fitting MiniBatchKMeans with {flat_features.shape[0]} samples and {self.kmeans_clusters} clusters...")
        self.kmeans.fit(flat_features)
        self.fitted = True
        self.logger.info("K-means fitting complete.")

        pseudo_labels_map = {}
        self.logger.info("Generating pseudo-labels from fitted K-means model...")
        for path in tqdm(audio_paths, desc="Predicting labels"):
            wav, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                wav = torchextra.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate) # Use torchextra.functional.resample if available, otherwise torchaudio.transforms.Resample
            wav = wav.to(device)

            extracted_feature = self._extract_features_from_audio(wav) # (B=1, T, D)
            z = self.kmeans.predict(extracted_feature.squeeze(0).cpu().numpy())  # (T,)

            fname = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(self.save_dir, f"{fname}.npy")
            np.save(out_path, z.astype(np.int32))
            
            pseudo_labels_map[path] = z.astype(np.int32)

        self.logger.info("Pseudo-label generation complete for current iteration.")
        return pseudo_labels_map

import os
import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
from typing import Dict, List, Literal, Optional
import logging
import torchaudio.transforms as T_audio # Import for Resample

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
        logger=None
    ):
        self.kmeans_clusters = kmeans_clusters
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        self.logger = logger if logger is not None else self._get_default_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        self.kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=1024, random_state=0, n_init='auto')
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        self.fitted = False

        self._current_input_type: Optional[Literal["mfcc", "transformer"]] = None
        self._current_model: Optional[torch.nn.Module] = None
        self._current_transformer_layer: Optional[int] = None

    def _get_default_logger(self):
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
        if is_mfcc:
            self._current_input_type = "mfcc"
            self._current_model = None
            self._current_transformer_layer = None
        else:
            if model is None:
                raise ValueError("Model must be provided for transformer-based feature extraction.")
            self._current_input_type = "transformer"
            self._current_model = model
            self._current_transformer_layer = transformer_layer

    def _extract_features_from_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if self._current_input_type == "mfcc":
            mfcc = self.mfcc_extractor(audio).transpose(1, 2)
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)
            return torch.cat([mfcc, delta, delta2], dim=-1)

        elif self._current_input_type == "transformer":
            if self._current_model is None:
                raise RuntimeError("Model not set for transformer-based feature extraction.")
            with torch.no_grad():
                feat = self._current_model.feature_extractor(audio)
                feat = feat.transpose(1, 2)
                enc = self._current_model.encoder.extract_layer(feat, self._current_transformer_layer)
            return enc
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
        self.set_current_iteration_mode(is_mfcc, model, transformer_layer)
        
        all_features_for_kmeans = []
        self.logger.info("Collecting features for K-means fitting...")
        for path in tqdm(audio_paths, desc="Extracting features"):
            wav, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                # Using torchaudio.transforms.Resample for resampling
                resampler = T_audio.Resample(orig_freq=sr, new_freq=self.sample_rate).to(device)
                wav = resampler(wav)
            wav = wav.to(device)
            
            extracted_feature = self._extract_features_from_audio(wav)
            all_features_for_kmeans.append(extracted_feature.squeeze(0).cpu().numpy())

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
                # Using torchaudio.transforms.Resample for resampling
                resampler = T_audio.Resample(orig_freq=sr, new_freq=self.sample_rate).to(device)
                wav = resampler(wav)
            wav = wav.to(device)

            extracted_feature = self._extract_features_from_audio(wav)
            z = self.kmeans.predict(extracted_feature.squeeze(0).cpu().numpy())

            fname = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(self.save_dir, f"{fname}.npy")
            np.save(out_path, z.astype(np.int32))
            
            pseudo_labels_map[path] = z.astype(np.int32)

        self.logger.info("Pseudo-label generation complete for current iteration.")
        return pseudo_labels_map
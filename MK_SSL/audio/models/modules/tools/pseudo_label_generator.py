# File: MK_SSL/audio/models/modules/tools.py

import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Literal, Optional, Dict, List
from torch.utils.data import DataLoader 
import logging

from MK_SSL.audio.models.modules.feature_extractors import MFCCFeatureExtractor


class PseudoLabelGenerator:
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
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        os.makedirs(self.save_dir, exist_ok=True)
        self.kmeans = MiniBatchKMeans(
            n_clusters=kmeans_clusters,
            batch_size=1024,
            random_state=0,
            n_init='auto'
        )
        self.fitted = False
        self.model = None
        self.layer = None
        self.device = None

    def _extract_features_for_clustering_batch(self, audio_batch: torch.Tensor, is_mfcc: bool) -> torch.Tensor:
        """Extract features for the entire batch (B, T) -> (B, T', D)."""
        audio_batch = audio_batch.to(self.device)
        with torch.no_grad():
            if is_mfcc:
                if hasattr(self.model, 'feature_extractor') and isinstance(self.model.feature_extractor, MFCCFeatureExtractor):
                    feats = self.model.feature_extractor(audio_batch)
                else:
                    self.logger.warning(
                        "Model's feature_extractor is not MFCC-based. Using temporary MFCCFeatureExtractor."
                    )
                    temp_mfcc_extractor = MFCCFeatureExtractor(sample_rate=self.sample_rate, n_mfcc=39).to(self.device)
                    feats = temp_mfcc_extractor(audio_batch)
            else:
                feats, _ = self.model.feature_extractor(audio_batch)
                feats = self.model.feature_projection(feats)
                feats = self.model.post_extract_proj_norm(feats)
                feats = self.model.post_extract_proj_dropout(feats)
                feats = self.model.encoder(feats)

            if isinstance(feats, tuple):  # If encoder returns (output, attn)
                feats = feats[0]
        return feats  # Shape: (B, T', D)

    def generate_pseudo_labels(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        is_mfcc: bool,
        transformer_layer: Optional[int],
        device: torch.device
    ) -> Dict[int, np.ndarray]:
        """Generate pseudo-labels for every dataset sample."""
        self.model = model.eval().to(device)
        self.layer = transformer_layer
        self.device = device

        kmeans_model_path = os.path.join(self.save_dir, "kmeans_model.pkl")

        # Pre-initialize dict
        dataset_len = len(dataloader.dataset)
        idx_to_labels = {i: None for i in range(dataset_len)}

        # === First Pass: Feature Extraction ===
        all_features_flattened = []
        all_indices = []

        self.logger.info(f"Starting feature extraction for K-means clustering on {dataset_len} samples...")
        for batch in tqdm(dataloader, desc="Feature Extraction (K-means)"):
            audio_batch = batch["audio"]
            indices_batch = batch["original_idx"]

            feats_batch = self._extract_features_for_clustering_batch(audio_batch, is_mfcc)
            feats_batch_np = feats_batch.cpu().numpy()

            for i, idx in enumerate(indices_batch):
                idx = int(idx)
                sample_feats = feats_batch_np[i]
                if sample_feats.shape[0] == 0:
                    self.logger.warning(f"Skipping index {idx}: No features extracted.")
                    continue

                all_features_flattened.append(sample_feats.reshape(-1, sample_feats.shape[-1]))
                all_indices.append(idx)
                idx_to_labels[idx] = sample_feats  # Temporarily store features

        # === Fit or Load K-means ===
        flat_features_for_kmeans = np.concatenate(all_features_flattened, axis=0)
        if not self.fitted:
            self.logger.info("Fitting K-means model...")
            if flat_features_for_kmeans.shape[0] < self.kmeans_clusters:
                self.logger.warning(
                    f"Samples ({flat_features_for_kmeans.shape[0]}) < clusters ({self.kmeans_clusters}). "
                    f"Reducing clusters to {flat_features_for_kmeans.shape[0]}."
                )
                self.kmeans = MiniBatchKMeans(
                    n_clusters=flat_features_for_kmeans.shape[0],
                    batch_size=1024,
                    random_state=0,
                    n_init='auto'
                )
            self.kmeans.fit(flat_features_for_kmeans)
            self.fitted = True
            self.logger.info(f"K-means fitted with {self.kmeans.n_clusters} clusters.")

            try:
                import joblib
                joblib.dump(self.kmeans, kmeans_model_path)
                self.logger.info(f"K-means model saved at {kmeans_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save K-means model: {e}")
        else:
            self.logger.info("Using pre-fitted K-means model...")
            if os.path.exists(kmeans_model_path):
                try:
                    import joblib
                    self.kmeans = joblib.load(kmeans_model_path)
                    self.logger.info(f"K-means model loaded from {kmeans_model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load K-means model: {e}")

        # === Second Pass: Label Assignment ===
        self.logger.info("Assigning pseudo-labels...")
        for idx in tqdm(all_indices, desc="Label Assignment"):
            sample_feats = idx_to_labels[idx]
            predicted_labels = self.kmeans.predict(sample_feats)
            idx_to_labels[idx] = predicted_labels

        # Safety check
        missing_indices = [i for i, v in idx_to_labels.items() if v is None]
        if missing_indices:
            raise RuntimeError(f"Missing pseudo-labels for {len(missing_indices)} samples: {missing_indices[:10]}...")

        self.logger.info("Pseudo-label generation completed.")
        return idx_to_labels

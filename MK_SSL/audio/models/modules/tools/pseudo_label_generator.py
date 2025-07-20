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
        logger=None # Add logger parameter
    ):
        self.kmeans_clusters = kmeans_clusters
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        self.logger = logger if logger is not None else logging._get_default_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        self.kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=1024, random_state=0, n_init='auto') # n_init='auto' is more robust
        self.fitted = False
        self.model = None # Model will be set during generate_pseudo_labels call
        self.layer = None # Layer will be set during generate_pseudo_labels call
        self.device = None # Device will be set during generate_pseudo_labels call

    def _extract_features_for_clustering(self, audio_tensor: torch.Tensor, is_mfcc: bool) -> torch.Tensor:
        """
        Extracts features (MFCC or Transformer's feature extractor output) from a single audio tensor.
        audio_tensor is assumed to be (T,) and will be unsqueezed to (1, T) for feature extractor.
        Returns (T_feat, D_feat) after squeezing the batch dimension.
        """
        # Add batch dimension as feature extractors expect (B, T) or (B, T, C)
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device) # (1, T)

        if is_mfcc:
            # If the model itself has an MFCC feature extractor configured (for iter 0)
            if hasattr(self.model, 'feature_extractor') and isinstance(self.model.feature_extractor, MFCCFeatureExtractor):
                feat = self.model.feature_extractor(audio_tensor) # (1, T_mfcc, D_mfcc)
            else:
                # Fallback: Create a temporary MFCC extractor if the model's isn't MFCC based.
                # This assumes standard MFCC parameters (e.g., from HuBERT paper).
                self.logger.warning("Model's feature_extractor is not MFCC-based for MFCC feature extraction. "
                                    "Using a temporary MFCCFeatureExtractor (n_mfcc=39, default sample_rate).")
                temp_mfcc_extractor = MFCCFeatureExtractor(
                    sample_rate=self.sample_rate,
                    n_mfcc=39 # Common default for HuBERT
                ).to(self.device)
                feat = temp_mfcc_extractor(audio_tensor) # (1, T_mfcc, D_mfcc)
            return feat.squeeze(0) # Returns (T_mfcc, D_mfcc)

        else: # Use model's ConvFeatureExtractor and Transformer encoder
            with torch.no_grad():

                conv_feats, _ = self.model.feature_extractor(audio_tensor)
                conv_feats = self.model.feature_projection(conv_feats)  # Project 512 -> 768
                conv_feats = self.model.post_extract_proj_norm(conv_feats)
                conv_feats = self.model.post_extract_proj_dropout(conv_feats)
                
                if self.layer is not None:
                    encoder_output = self.model.encoder.extract_layer(conv_feats, self.layer)  
                else: # Default to last layer if no specific layer provided
                    encoder_output = self.model.encoder(conv_feats)
                    if not isinstance(encoder_output, torch.Tensor):
                         # Handle cases where encoder might return tuple (output, attn_weights)
                         encoder_output = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                    self.logger.debug(f"Encoder output shape without specific layer: {encoder_output.shape}")

            return encoder_output.squeeze(0) # Returns (T', D)

    def generate_pseudo_labels(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        is_mfcc: bool,
        transformer_layer: Optional[int],
        device: torch.device
    ) -> Dict[int, np.ndarray]:
        self.model = model.eval().to(device)
        self.layer = transformer_layer
        self.device = device

        # Cache paths
        kmeans_model_path = os.path.join(self.save_dir, "kmeans_model.pkl")

        # First pass: collect all features
        all_features_flattened = []
        sample_features_list = []
        original_indices_collected = []

        self.logger.info("Starting feature extraction for K-means clustering (first pass over data)...")
        for batch in tqdm(dataloader, desc="Feature Extraction (K-means)"):
            audio_batch = batch["audio"]  # (B, T)
            indices_batch = batch["original_idx"]  # (B,)

            batch_features_list = []
            for i in range(audio_batch.shape[0]):
                single_audio = audio_batch[i]
                feat = self._extract_features_for_clustering(single_audio, is_mfcc)
                batch_features_list.append(feat.cpu().numpy())
                all_features_flattened.append(feat.cpu().numpy().reshape(-1, feat.shape[-1]))

            sample_features_list.extend(batch_features_list)
            original_indices_collected.extend(indices_batch.cpu().tolist())

        # Fit or load K-means
        if not self.fitted:
            self.logger.info("Fitting K-means model...")
            flat_features_for_kmeans = np.concatenate(all_features_flattened, axis=0)
            if flat_features_for_kmeans.shape[0] < self.kmeans_clusters:
                self.logger.warning(
                    f"Number of samples ({flat_features_for_kmeans.shape[0]}) < n_clusters ({self.kmeans_clusters}). "
                    f"Reducing n_clusters to {flat_features_for_kmeans.shape[0]}."
                )
                self.kmeans = MiniBatchKMeans(
                    n_clusters=flat_features_for_kmeans.shape[0],
                    batch_size=1024,
                    random_state=0,
                    n_init='auto'
                )
            self.kmeans.fit(flat_features_for_kmeans)
            self.fitted = True
            self.logger.info(f"K-means clustering completed with {self.kmeans.n_clusters} clusters.")

            # Save model
            try:
                import joblib
                joblib.dump(self.kmeans, kmeans_model_path)
                self.logger.info(f"K-means model saved at {kmeans_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save K-means model: {e}")
        else:
            self.logger.info("Using pre-fitted K-means model for pseudo-label generation.")
            if os.path.exists(kmeans_model_path):
                try:
                    import joblib
                    self.kmeans = joblib.load(kmeans_model_path)
                    self.logger.info(f"K-means model loaded from {kmeans_model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load K-means model: {e}")

        # Second pass: generate labels
        self.logger.info("Generating pseudo-labels from fitted K-means model (second pass over features)...")
        idx_to_labels = {}
        for i, original_idx in enumerate(original_indices_collected):
            sample_feats = sample_features_list[i]
            predicted_labels = self.kmeans.predict(sample_feats)
            idx_to_labels[original_idx] = predicted_labels

        self.logger.info("Pseudo-label generation completed.")
        return idx_to_labels

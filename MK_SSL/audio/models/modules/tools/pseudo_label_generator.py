# File: MK_SSL/audio/models/modules/tools.py

import os
import numpy as np
import torch
import torchaudio # Still needed for potential raw audio loading if not coming from DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Literal, Optional, Dict, List
from torch.utils.data import DataLoader # New import for DataLoader type hint
import logging

# Assuming these are available, if not, they need to be defined or provided
from MK_SSL.audio.models.modules.feature_extractors import MFCCFeatureExtractor, ConvFeatureExtractor
# Note: Ensure MFCCFeatureExtractor and ConvFeatureExtractor are correctly imported or defined.


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
                # self.model.feature_extractor is ConvFeatureExtractor here
                conv_feats = self.model.feature_extractor(audio_tensor)  # (1, C, T')
                conv_feats = conv_feats.transpose(1, 2)  # (1, T', C)

                if self.layer is not None:
                    encoder_output = self.model.encoder.extract_layer(conv_feats, self.layer)  # (1, T', D)
                else: # Default to last layer if no specific layer provided
                    encoder_output = self.model.encoder.forward_features(conv_feats) # Assuming encoder has a forward_features or similar
                    if not isinstance(encoder_output, torch.Tensor):
                         # Handle cases where encoder might return tuple (output, attn_weights)
                         encoder_output = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                    self.logger.debug(f"Encoder output shape without specific layer: {encoder_output.shape}")

            return encoder_output.squeeze(0) # Returns (T', D)

    def generate_pseudo_labels(
        self,
        dataloader: DataLoader, # Now takes a DataLoader (wrapping HuBERTWrapperDataset)
        model: torch.nn.Module,
        is_mfcc: bool,
        transformer_layer: Optional[int],
        device: torch.device
    ) -> Dict[int, np.ndarray]: # Returns dict mapping original_idx to labels
        self.model = model.eval().to(device) # Ensure model is in eval mode and on device
        self.layer = transformer_layer
        self.device = device

        all_features_flattened = [] # For K-means fitting (all time steps concatenated)
        sample_features_list = [] # List of (T_feat, D_feat) numpy arrays for each sample
        original_indices_collected = [] # Corresponding original indices for each sample

        self.logger.info("Starting feature extraction for K-means clustering (first pass over data)...")
        # Ensure the dataloader provides {"audio": audio_tensor, "original_idx": idx} from HuBERTWrapperDataset
        for batch in tqdm(dataloader, desc="Feature Extraction (K-means)"):
            audio_batch = batch["audio"] # (B, T)
            indices_batch = batch["original_idx"] # (B,)

            # Extract features for each sample in the batch
            batch_features_list = []
            for i in range(audio_batch.shape[0]):
                single_audio = audio_batch[i] # Get (T,) tensor for single sample
                feat = self._extract_features_for_clustering(single_audio, is_mfcc) # Returns (T_feat, D_feat)
                
                batch_features_list.append(feat.cpu().numpy()) # Store numpy array of (T_feat, D_feat)
                all_features_flattened.append(feat.cpu().numpy().reshape(-1, feat.shape[-1])) # Flatten all time steps for K-means fit

            sample_features_list.extend(batch_features_list)
            original_indices_collected.extend(indices_batch.cpu().tolist())

        # Fit K-means if not already fitted
        if not self.fitted:
            self.logger.info("Fitting K-means model...")
            flat_features_for_kmeans = np.concatenate(all_features_flattened, axis=0)
            self.kmeans.fit(flat_features_for_kmeans)
            self.fitted = True
            self.logger.info(f"K-means clustering completed with {self.kmeans.n_clusters} clusters.")
        else:
            self.logger.info("Using pre-fitted K-means model for pseudo-label generation.")

        # Generate labels for all features, mapping back to original_idx
        self.logger.info("Generating pseudo-labels from fitted K-means model (second pass over features)...")
        idx_to_labels = {}
        for i, original_idx in enumerate(original_indices_collected):
            sample_feats = sample_features_list[i] # (T_feat, D_feat)
            predicted_labels = self.kmeans.predict(sample_feats) # (T_feat,)
            idx_to_labels[original_idx] = predicted_labels # Map original_idx to its label sequence

        self.logger.info("Pseudo-label generation completed.")
        return idx_to_labels
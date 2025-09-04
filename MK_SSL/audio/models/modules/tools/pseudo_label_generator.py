
import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Optional, Dict
from torch.utils.data import DataLoader 
import logging
import joblib

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

    def _extract_features_for_clustering_batch(
        self,
        audio_batch: torch.Tensor,
        is_mfcc: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features for K-means.
        - Iter 0: MFCCs
        - Iter >=1: Transformer *layer* hidden states (per HuBERT paper).
        Returns (B, T', D).
        """
        audio_batch = audio_batch.to(self.device)
        lengths = lengths.to(self.device) if lengths is not None else None

        with torch.no_grad():
            if is_mfcc:
                # Iteration 0: MFCC features
                if hasattr(self.model, "feature_extractor") and \
                self.model.feature_extractor.__class__.__name__ == "MFCCFeatureExtractor":
                    feats = self.model.feature_extractor(audio_batch)  # (B, T', D)
                    # MFCC extractor generally doesn't return lengths; that's fine for KMeans
                else:
                    from MK_SSL.audio.models.modules.feature_extractors import MFCCFeatureExtractor
                    temp_mfcc_extractor = MFCCFeatureExtractor(
                        sample_rate=self.sample_rate,
                        n_mfcc=39
                    ).to(self.device)
                    feats = temp_mfcc_extractor(audio_batch)  # (B, T', D)
            else:
                # Iterations >= 1: conv -> proj -> specific transformer layer
                feats, out_lengths = self.model.feature_extractor(audio_batch, lengths)  # (B, T', C), (B,)
                feats = self.model.feature_projection(feats)
                feats = self.model.post_extract_proj_norm(feats)
                feats = self.model.post_extract_proj_dropout(feats)

                use_layer = self.layer  # set in generate_pseudo_labels(...)
                if use_layer is None or use_layer <= 0:
                    # Fallback to final layer if not specified
                    feats = self.model.encoder(feats, out_lengths)  # final layer output
                else:
                    # Critical fix: take the requested layer's hidden states
                    feats = self.model.encoder.extract_layer(
                        feats, layer=int(use_layer), lengths=out_lengths
                    )  # (B, T', D)

        return feats  # (B, T', D)



    def generate_pseudo_labels(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        is_mfcc: bool,
        transformer_layer: Optional[int],
        device: torch.device,
        iteration_id: Optional[int] = None,  # NEW
    ) -> Dict[int, np.ndarray]:
        """
        Generate pseudo-labels for every dataset sample.
    
        Args:
            dataloader (DataLoader): DataLoader for feature extraction (no shuffling).
            model (torch.nn.Module): HuBERT model used for feature extraction.
            is_mfcc (bool): Whether to extract MFCC features for the first iteration.
            transformer_layer (Optional[int]): Specific transformer layer to use.
            device (torch.device): Device to perform computations on.
            iteration_id (Optional[int]): HuBERT iteration number to version KMeans per-iter.
    
        Returns:
            Dict[int, np.ndarray]: Mapping from dataset indices to pseudo-label sequences.
        """
        self.model = model.eval().to(device)
        self.layer = transformer_layer
        self.device = device
    
        dataset_len = len(dataloader.dataset)
        idx_to_labels = {i: None for i in range(dataset_len)}
        seen_indices = set()
    
        self.logger.info(f"Starting feature extraction for K-means clustering on {dataset_len} samples...")
        all_features_flattened = []
        all_indices = []
    
        for batch in tqdm(dataloader, desc="Feature Extraction (K-means)"):
            audio_batch = batch["audio"]
            indices_batch = batch["original_idx"].tolist()
            lengths_batch = batch.get("length", None)


            feats_batch = self._extract_features_for_clustering_batch(
                audio_batch=audio_batch,
                is_mfcc=is_mfcc,
                lengths=lengths_batch
            )
            if isinstance(feats_batch, tuple):  # guard
                feats_batch = feats_batch[0]
    
            feats_batch_np = feats_batch.detach().cpu().numpy()
    
            for i, idx in enumerate(indices_batch):
                idx = int(idx)
                if idx in seen_indices:
                    self.logger.warning(f"[WARNING] Duplicate dataset index {idx} encountered. Skipping duplicate.")
                    continue
                if idx >= dataset_len:
                    self.logger.warning(f"[WARNING] Invalid index {idx} (out of range). Skipping.")
                    continue
                
                sample_feats = feats_batch_np[i]
                if sample_feats.shape[0] == 0:
                    self.logger.warning(f"Skipping index {idx}: No features extracted.")
                    continue
                
                seen_indices.add(idx)
                flat_feats = sample_feats.reshape(-1, sample_feats.shape[-1])  # (T', D)
                all_features_flattened.append(flat_feats)
                all_indices.append(idx)
                idx_to_labels[idx] = flat_feats
    
        # Concatenate all features for KMeans fitting
        if len(all_features_flattened) == 0:
            raise ValueError("No features were extracted for KMeans fitting.")
        flat_features_for_kmeans = np.concatenate(all_features_flattened, axis=0)  # (N_total, D)
        feature_dim = int(flat_features_for_kmeans.shape[1])
    
        # Version the KMeans model by iteration and feature dim
        iter_tag = f"iter_{iteration_id if iteration_id is not None else 0}"
        kmeans_model_path = os.path.join(self.save_dir, f"kmeans_{iter_tag}_d{feature_dim}.pkl")
    
        # Decide whether to (re)fit or load
        need_refit = True
        if os.path.exists(kmeans_model_path):
            try:
                loaded = joblib.load(kmeans_model_path)
                # Check dimension compatibility
                if getattr(loaded, "n_features_in_", None) == feature_dim:
                    self.kmeans = loaded
                    need_refit = False
                    self.logger.info(f"Loaded KMeans from {kmeans_model_path} (D={feature_dim}).")
                else:
                    self.logger.info(
                        f"Existing KMeans has n_features_in_={getattr(loaded, 'n_features_in_', None)} "
                        f"but current features have D={feature_dim}. Will refit."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load KMeans from {kmeans_model_path}: {e}. Will refit.")
    
        if need_refit:
            n_clusters = self.kmeans_clusters
            if flat_features_for_kmeans.shape[0] < n_clusters:
                self.logger.warning(
                    f"Samples ({flat_features_for_kmeans.shape[0]}) < clusters ({n_clusters}). "
                    f"Reducing clusters to {flat_features_for_kmeans.shape[0]}."
                )
                n_clusters = flat_features_for_kmeans.shape[0]
    
            self.kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                random_state=0,
                n_init='auto'
            )
            self.logger.info(f"Fitting KMeans (k={n_clusters}, D={feature_dim})...")
            self.kmeans.fit(flat_features_for_kmeans)
            try:
                joblib.dump(self.kmeans, kmeans_model_path)
                self.logger.info(f"KMeans saved at {kmeans_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save KMeans model: {e}")
    
        # Assign labels per sample
        self.logger.info("Assigning pseudo-labels...")
        for idx in tqdm(all_indices, desc="Label Assignment"):
            sample_feats = idx_to_labels[idx]
            # scikit-learn requires 2D input
            if sample_feats.ndim != 2 or sample_feats.shape[1] != feature_dim:
                raise ValueError(
                    f"Sample idx {idx}: feature shape {sample_feats.shape} incompatible with KMeans D={feature_dim}"
                )
            predicted_labels = self.kmeans.predict(sample_feats)
            idx_to_labels[idx] = predicted_labels.astype(np.int64)
    
        # Fill missing/extra indices safely
        all_dataset_indices = set(range(dataset_len))
        missing_indices = all_dataset_indices - seen_indices
        extra_indices = seen_indices - all_dataset_indices
    
        if extra_indices:
            self.logger.warning(f"Found {len(extra_indices)} extra indices: {sorted(list(extra_indices))[:10]}...")
            for idx in extra_indices:
                idx_to_labels.pop(idx, None)
    
        if missing_indices:
            self.logger.warning(f"Filling {len(missing_indices)} missing indices with a single zero label.")
            zero_label = np.zeros((1,), dtype=np.int64)
            for idx in missing_indices:
                idx_to_labels[idx] = zero_label
    
        self.logger.info(f"Pseudo-label generation completed with {len(idx_to_labels)} samples.")
        return idx_to_labels
    
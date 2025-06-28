import os
import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Literal, Optional


class PseudoLabelGenerator:
    """
    Generates HuBERT pseudo-labels (hidden units) from MFCCs or Transformer layer outputs.

    Args:
        input_type (Literal["mfcc", "transformer"]): Feature source type.
        model (Optional[torch.nn.Module]): HuBERT model for transformer-based feature extraction.
        transformer_layer (Optional[int]): Transformer layer index to extract features from.
        sample_rate (int): Audio sampling rate.
        kmeans_clusters (int): Number of clusters for K-means.
        save_dir (str): Path to save generated labels.
    """

    def __init__(
        self,
        input_type: Literal["mfcc", "transformer"],
        model: Optional[torch.nn.Module] = None,
        transformer_layer: Optional[int] = None,
        sample_rate: int = 16000,
        kmeans_clusters: int = 100,
        save_dir: str = "generated_labels",
    ):
        self.input_type = input_type
        self.model = model.eval() if model else None
        self.layer = transformer_layer
        self.kmeans_clusters = kmeans_clusters
        self.sample_rate = sample_rate
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=1024, random_state=0)
        self.fitted = False

        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        if self.input_type == "mfcc":
            mfcc = self.mfcc_extractor(audio).transpose(1, 2)  # (B, T, 13)
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)
            return torch.cat([mfcc, delta, delta2], dim=-1).squeeze(0)  # (T, 39)

        elif self.input_type == "transformer":
            with torch.no_grad():
                feat = self.model.feature_extractor(audio)  # (B, C, T')
                feat = feat.transpose(1, 2)  # (B, T', C)
                enc = self.model.encoder.extract_layer(feat, self.layer)  # (B, T', D)
            return enc.squeeze(0)

        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

    def fit_kmeans(self, audio_paths: list):
        all_features = []
        for path in tqdm(audio_paths, desc="Extracting features for k-means"):
            wav, sr = torchaudio.load(path)
            assert sr == self.sample_rate
            x = self.extract_features(wav)
            all_features.append(x.cpu().numpy())

        flat = np.concatenate(all_features, axis=0)
        self.kmeans.fit(flat)
        self.fitted = True

    def generate_labels(self, audio_paths: list):
        if not self.fitted:
            raise RuntimeError("You must call fit_kmeans() before generate_labels().")

        for path in tqdm(audio_paths, desc="Generating pseudo-labels"):
            wav, sr = torchaudio.load(path)
            assert sr == self.sample_rate
            x = self.extract_features(wav)  # (T, D)
            z = self.kmeans.predict(x.cpu().numpy())  # (T,)

            fname = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(self.save_dir, f"{fname}.npy")
            np.save(out_path, z.astype(np.int32))

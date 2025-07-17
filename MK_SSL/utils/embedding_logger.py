import os
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px


class EmbeddingLogger:
    def __init__(
        self,
        log_dir: str,
        method_name: str,
        reduce_method: str = "tsne",
        save_embeddings: bool = True,
        log_interval: int = 1,
    ):
        self.log_dir = os.path.join(log_dir, method_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.reduce_method = reduce_method
        self.save_embeddings = save_embeddings
        self.log_interval = log_interval
        self.embeddings = []
        self.labels = []
        self.steps = []

    def log_step(self, step: int, embeddings: torch.Tensor, labels: torch.Tensor):
        if step % self.log_interval != 0:
            return

        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()

        self.embeddings.append(embeddings)
        self.labels.append(labels)
        self.steps.append(step)

        if self.save_embeddings:
            torch.save(
                {"embeddings": embeddings, "labels": labels},
                os.path.join(self.log_dir, f"step_{step}.pt"),
            )

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        if self.reduce_method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
        return reducer.fit_transform(features)

    def plot_all(self):
        records = []

        for step, embeddings, labels in zip(self.steps, self.embeddings, self.labels):
            embeddings_np = embeddings.numpy()
            labels_np = labels.numpy()
            reduced = self._reduce(embeddings_np)

            df = pd.DataFrame(reduced, columns=["x", "y"])
            df["label"] = labels_np.astype(str)
            df["step"] = step
            records.append(df)

        full_df = pd.concat(records, ignore_index=True)

        fig = px.scatter(
            full_df,
            x="x", y="y",
            color="label",
            animation_frame="step",
            title="2D Embedding Animation",
            opacity=0.6,
            height=700,
            width=900
        )

        save_path = os.path.join(self.log_dir, "embedding_animation.html")
        fig.write_html(save_path)
        return save_path

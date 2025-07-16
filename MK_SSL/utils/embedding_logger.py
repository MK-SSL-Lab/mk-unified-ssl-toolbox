import os
import torch
import numpy as np
from typing import List, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # for headless environments

class EmbeddingLogger:
    def __init__(self, log_dir: str, method_name: str, reduce_method: str = "tsne", save_embeddings: bool = True):
        self.log_dir = os.path.join(log_dir, method_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.reduce_method = reduce_method
        self.save_embeddings = save_embeddings

        self.embeddings = []
        self.labels = []
        self.steps = []

    def log_step(self, step: int, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Log embedding step with labels.

        Args:
            step (int): Training step or epoch.
            embeddings (Tensor): (B, D)
            labels (Tensor): (B,)
        """
        embeddings = embeddings.detach().cpu()
        labels = labels.detach().cpu()

        self.embeddings.append(embeddings)
        self.labels.append(labels)
        self.steps.append(step)

        if self.save_embeddings:
            torch.save(
                {"embeddings": embeddings, "labels": labels},
                os.path.join(self.log_dir, f"step_{step}.pt")
            )

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        if self.reduce_method == "pca":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
        return reducer.fit_transform(features)

    def plot_step(self, step: int):
        embeddings = self.embeddings[self.steps.index(step)].numpy()
        labels = self.labels[self.steps.index(step)].numpy()

        reduced = self._reduce(embeddings)

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", loc="best")
        plt.title(f"Step {step} Embedding Visualization")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"step_{step}_plot.png"))
        plt.close()

    def plot_all(self):
        for step in self.steps:
            self.plot_step(step)

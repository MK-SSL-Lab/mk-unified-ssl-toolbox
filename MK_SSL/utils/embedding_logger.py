import os
import torch
import numpy as np
from typing import Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # For headless environments

class EmbeddingLogger:
    def __init__(
        self,
        log_dir: str,
        method_name: str,
        reduce_method: str = "tsne",
        save_embeddings: bool = True,
        log_interval: int = 1  # NEW: log every N steps
    ):
        self.log_dir = os.path.join(log_dir, method_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.reduce_method = reduce_method
        self.save_embeddings = save_embeddings
        self.log_interval = log_interval

        self.embeddings = []
        self.steps = []

    def log_step(self, step: int, embeddings: torch.Tensor):
        """
        Logs embeddings at a specific step (if it matches interval).
        """
        if step % self.log_interval != 0:
            return

        embeddings = embeddings.detach().cpu()
        self.embeddings.append(embeddings)
        self.steps.append(step)

        if self.save_embeddings:
            torch.save(
                {"embeddings": embeddings},
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
        reduced = self._reduce(embeddings)

        plt.figure(figsize=(6, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
        plt.title(f"Step {step} Embedding Visualization")
        plt.tight_layout()
        path = os.path.join(self.log_dir, f"step_{step}_plot.png")
        plt.savefig(path)
        plt.close()
        return path

    def plot_all(self):
        for step in self.steps:
            self.plot_step(step)

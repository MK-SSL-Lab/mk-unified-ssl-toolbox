<p align="center">
  <img src="logo2.png" alt="MK_SSL Logo" width="50%"/>
</p>

<h1 align="center">
  <br>MK_SSL: A Modular Self-Supervised Learning Library with High-Level API for Audio, Vision, Graph, and Cross-Modal Data
</h1>

<p align="center">
  <img alt="License" src="https://img.shields.io/github/license/kia-vadaei/MK_SSL?color=blue"/>
  <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"/>
  <img alt="PyPI" src="https://img.shields.io/pypi/v/mk-ssl"/>
</p>

---

## 📚 Table of Contents
- [📍 Overview](#-overview)
- [🔧 Installation](#-installation)
- [🧠 Supported Methods](#-supported-methods)
- [💻 Tutorial](#-tutorial)
- [📊 Benchmarks](#-benchmarks)
- [🛠️ Additional Tools Used](#️-additional-tools-used)
- [💡 License](#-license)
- [👥 Collaborators](#-collaborators)

---

## 📍 Overview
**MK_SSL** is a modular, extensible, and high-level self-supervised learning library supporting a wide range of modalities:

- 🎧 **Audio**
- 🖼️ **Vision**
- 🌐 **Graph**
- 🔀 **Cross-Modal** (e.g. audio-text, audio-image)

Designed to empower researchers, engineers, and learners, **MK_SSL** brings unified training, evaluation, visualization, and backbone export for a wide set of cutting-edge SSL methods with minimum effort.

> No thesis tag. No university mention. Just clean, production-level design.

---

## 🔧 Installation
```bash
pip install mk-ssl
```

---

## 🧠 Supported Methods
Methods are grouped by modality. Each is implemented modularly and documented with help accessible via Python’s `help()`.

### 🎧 Audio
<details><summary><b>Wav2Vec2</b></summary>
A contrastive model leveraging a CNN feature encoder and transformer context network. It masks latent speech representations and trains via contrastive loss over negatives.

- **Loss**: Contrastive
- **Backbone**: CNN + Transformer
- **Paper**: [Link](https://arxiv.org/abs/2006.11477)
</details>

<details><summary><b>HuBERT</b></summary>
A mask prediction model that uses k-means clustering on MFCCs to bootstrap hidden units, later replaced by predicted features.

- **Loss**: Cross-entropy on pseudo-labels
- **Backbone**: Transformer
- **Paper**: [Link](https://arxiv.org/abs/2106.07447)
</details>

<details><summary><b>COLA</b></summary>
Contrastive learning for audio using multi-view augmentations and InfoNCE. Offers optional projection heads and cosine similarity loss.

- **Loss**: InfoNCE
- **Backbone**: CNN
- **Paper**: [Link](https://arxiv.org/abs/2202.04539)
</details>

<details><summary><b>EAT</b></summary>
An augmentation-free framework that uses attention pooling and mean pooling alignment for learning universal audio representations.

- **Loss**: Contrastive
- **Backbone**: Conformer
- **Paper**: [Link](https://arxiv.org/abs/2203.12347)
</details>

<details><summary><b>SpeechSimCLR</b></summary>
Applies SimCLR to speech with log-mel features and audio augmentations. Augments pairs and minimizes NT-Xent loss.

- **Loss**: NT-Xent
- **Paper**: [Link](https://arxiv.org/abs/2005.09844)
</details>

### 🖼️ Vision
<details><summary><b>MAE (Masked Autoencoder)</b></summary>
Self-supervised pretraining using patch-masked reconstruction with ViTs.

- **Loss**: MSE Reconstruction
- **Backbone**: Vision Transformer (ViT)
- **Paper**: [Link](https://arxiv.org/abs/2111.06377)
</details>

### 🌐 Graph
<details><summary><b>GraphCL</b></summary>
A contrastive method for graph-level representation learning using data augmentations like node dropping, subgraph sampling, and edge perturbation.

- **Loss**: NT-Xent
- **Backbone**: GCN/GAT/GIN
- **Paper**: [Link](https://arxiv.org/abs/2010.13902)
</details>

### 🔀 Cross-Modal
<details><summary><b>CLAP</b></summary>
Contrastive Language-Audio Pretraining using paired audio and text data. Combines encoders for each modality to learn aligned representations.

- **Loss**: Contrastive
- **Paper**: [Link](https://arxiv.org/abs/2301.12667)
</details>

<details><summary><b>AudioCLIP</b></summary>
Extension of CLIP to audio-image-text tri-modal learning. Joint embedding with shared space optimization.

- **Loss**: Multi-modal contrastive
- **Paper**: [Link](https://arxiv.org/abs/2101.10249)
</details>

<details><summary><b>Wav2CLIP</b></summary>
A simple and effective model that maps audio to CLIP’s text-image embedding space.

- **Loss**: MSE
- **Paper**: [Link](https://arxiv.org/abs/2110.11499)
</details>

---

## 💻 Tutorial
Here’s how easy it is to use `MK_SSL` for training and evaluation:

```python
from MK_SSL.audio import Trainer

trainer = Trainer(
    method='wav2vec2',
    save_dir='./save_dir',
    checkpoint_interval=50,
    use_data_parallel=True,
    mixed_precision_training=False,
    wandb_mode='online',
    wandb_project='wav2vec2-pretext',
    wandb_run_name='run-librispeech',
    verbose=True,
    kwargs=kwargs,
)

trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    start_epoch=0,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-2,
    optimizer="adamw",
    use_hpo=True,
    n_trials=20,
    tuning_epochs=5,
    use_embedding_logger=True,
    logger_loader=logger_loader,
    kwargs=kwargs,
)

trainer.evaluate(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    num_classes=39,
    batch_size=64,
    lr=1e-3,
    epochs=10,
    freeze_backbone=True,
    kwargs=kwargs,
)

trainer.save_backbone()
```

Access full documentation with:
```python
help(trainer)
```

---

## 📊 Benchmarks

### 🎧 Wav2Vec2 (Audio)
| Dataset | Task | Model | Accuracy |
|---------|------|--------|----------|
| [LibriSpeech](#) | Speaker ID | Wav2Vec2 | 89.7% |

<p align="center">
  <img src="path/to/wav2vec2_vis1.png" width="32%"/>
  <img src="path/to/wav2vec2_vis2.png" width="32%"/>
  <img src="path/to/wav2vec2_vis3.png" width="32%"/>
</p>

---

### 🔀 Wav2CLIP (Cross-Modal)
<p align="center">
  <img src="path/to/wav2clip_visualization.png" width="80%"/>
</p>

---

### 🖼️ MAE (Vision)
| Dataset | Evaluation | Accuracy |
|---------|------------|----------|
| ImageNet-500 | Fine-tune | 81.2% |
| ImageNet-500 | Linear Probe | 70.5% |

---

### 🌐 GraphCL (Graph)
| Dataset | Task | Accuracy |
|---------|------|----------|
| BBBP | Toxicity | 72.4% |
| Tox21 | 12-Task Multi-Label | 77.9% |

<p align="center">
  <img src="path/to/graph_cluster_plot.png" width="80%"/>
</p>

---

## 🛠️ Additional Tools Used

✅ **LoRA (Low-Rank Adaptation)**: Enables lightweight fine-tuning for SSL models.

✅ **HuggingFace Support**: Seamless integration with `transformers` for NLP and audio.

✅ **WandB Integration**: Experiment tracking, visualization, logging.

✅ **Hyperparameter Optimization (HPO)**: Easily switch on with `use_hpo=True`, supports Optuna.

✅ **DDP (DistributedDataParallel)**: Scales your training to multiple GPUs effortlessly.

✅ **Dynamic Logging + Visualizations**: Use `use_embedding_logger=True` to enable clustering plots.

✅ **Mixed Precision Training**: Toggle with `mixed_precision_training=True`.

MK_SSL comes equipped with everything needed for **serious experimentation** — from **minimal research setups** to **fully scalable training pipelines**.

---

## 💡 License
This project is licensed under the [MIT License](./LICENSE).

---

## 👥 Collaborators
- [Melika Shirian](https://github.com/MelikaShirian12)
- [Kianoosh Vadaei](https://github.com/kia-vadaei)

### 🎓 Advisors
- [Dr. Peyman Adibi](https://scholar.google.com/citations?user=u-FQZMkAAAAJ)
- [Dr. Hossein Karshenas](https://scholar.google.com/citations?user=BjMFkWEAAAAJ)

---

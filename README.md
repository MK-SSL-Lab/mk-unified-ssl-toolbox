<p align="center">
  <img src="logo2.png" alt="MK_SSL Logo" width="50%"/>
</p>

<h1>
<br>MK_SSL: A Modular Self-Supervised Learning Library
</h1>

![GitHub](https://img.shields.io/github/license/kia-vadaei/MK_SSL) ![PyPI - Version](https://img.shields.io/pypi/v/mk-ssl) ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

---

## 📂 Table of Contents
- [📍 Overview](#-overview)
- [✍️ Self-Supervised Learning](#✍️-self-supervised-learning)
- [🔎 Supported Methods](#-supported-methods)
- [📦 Installation](#-installation)
- [💡 Tutorial](#-tutorial)
- [📊 Benchmarks](#-benchmarks)
- [⚖️ License](#-license)
- [🤝 Collaborators](#-collaborators)

---

## 📍 Overview

**MK_SSL** is a modular, extensible self-supervised learning (SSL) library designed to support diverse data modalities: **Audio, Vision, Graph**, and **Cross-Modal**. It comes with:

- High-level Trainer API
- Multi-method implementations
- Logging, visualization, and experiment management
- Lightweight and customizable integrations

This repository provides a toolkit for researchers and practitioners to develop, pretrain, evaluate, and visualize SSL models across domains.

---

## ✍️ Self-Supervised Learning

Self-supervised learning is a powerful training paradigm that eliminates the need for manual labels by formulating **pretext tasks** from the input data itself. These pretext-trained models can be fine-tuned on small labeled datasets for downstream tasks. MK_SSL enables seamless experimentation with SSL algorithms, modular components, and evaluation pipelines.

---

## 🔎 Supported Methods

### 🎧 Audio
- **Wav2Vec2**: Predicts quantized future audio representations.
- **HuBERT**: Learns discrete units via clustering of MFCCs or embeddings.
- **COLA**: Contrastive Learning with Adaptive Clustering.
- **EAT**: Emphasized Alignment Training with pair-level contrast.
- **SpeechSimCLR**: A SimCLR variant adapted for speech waveform augmentations.

### 🎨 Vision
- **MAE (Masked AutoEncoder)**: Learns visual representations by reconstructing masked image patches.

### 📈 Graph
- **GraphCL**: Contrastive learning for graphs using subgraph and edge augmentations.

### 📹 Cross-Modal
- **CLAP**: Audio-Language pretraining with contrastive loss.
- **AudioCLIP**: Trains audio encoder to match vision-language representations.
- **Wav2CLIP**: Maps raw waveforms to CLIP’s latent embedding space.

---

## 📦 Installation

Install MK_SSL from PyPI:

```bash
pip install mk-ssl
```

---

## 💡 Tutorial

Here is a minimal usage example for **Audio** models (e.g., Wav2Vec2):

### ➕ Initialization
```python
from MK_SSL.audio import Trainer

trainer = Trainer(
    method = "wav2vec2",
    save_dir = "./",
    checkpoint_interval = 50,
    use_data_parallel = True,
    variant = "base",
    reload_checkpoint = False,
    mixed_precision_training = False,
    verbose = True,
    wandb_mode = "online",
    wandb_run_name = "run-1-on-librispeech",
    kwargs=kwargs,
)
```

### ⚖️ Training
```python
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    start_epoch=0,
    epochs=100,
    optimizer="adamw",
    weight_decay=1e-2,
    lr=1e-4,
    use_hpo=True,
    n_trials=20,
    tuning_epochs=5,
    use_embedding_logger=True,
    logger_loader=logger_loader,
    kwargs=kwargs,
)
```

### 🔢 Evaluation
```python
trainer.evaluate(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    num_classes=39,
    batch_size=64,
    lr=1e-3,
    epochs=10,
    freeze_backbone=True,
    kwargs=kwargs
)
```

### 🔖 Save Pretrained Backbone
```python
trainer.save_backbone()
```

---

## 📊 Benchmarks

### 🎧 Audio - Wav2Vec2
| Task       | Dataset     | Model     | Accuracy |
|------------|-------------|-----------|----------|
| Keyword Spotting | SpeechCommands | Wav2Vec2  | 91.4%    |

**Latent Space Visualizations:**
<p align="center">
  <img src="wav2vec2_plot1.png" width="30%">
  <img src="wav2vec2_plot2.png" width="30%">
  <img src="wav2vec2_plot3.png" width="30%">
</p>

---

### 📹 Cross-Modal - Wav2CLIP
<p align="center">
  <img src="wav2clip_visualization1.png" width="40%">
  <img src="wav2clip_visualization2.png" width="40%">
</p>

---

### 🎨 Vision - MAE (ImageNet-500)
| Model | Probe Type     | Accuracy |
|--------|----------------|----------|
| MAE    | Fine-tuned     | 83.2%    |
| MAE    | Linear         | 67.5%    |

---

### 📈 Graph - GraphCL
| Task   | Dataset | Accuracy |
|--------|---------|----------|
| FT     | BBBP    | 73.6%    |

| Task         | Dataset | Target | Accuracy |
|--------------|---------|--------|----------|
| FT           | Tox21   | Task 1 | 78.2%    |
|              |         | Task 2 | 76.1%    |
|              |         | ...    | ...      |
|              |         | Task 12| 79.3%    |

<p align="center">
  <img src="graph_result.png" width="60%" />
</p>

---

## ⚖️ License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Collaborators

Developed by:
- [Melika Shirian](https://github.com/MelikaShirian12)
- [Kianoosh Vadaei](https://github.com/kia-vadaei)

With gratitude to our advisors:
- Dr. Peyman Adibi
- Dr. Hossein Karshenas

---

## ⚡️ Additional Tools Used
- ✅ Distributed Training (DDL)
- ⚖️ Hyperparameter Optimization (HPO)
- 🔊 Logging & Monitoring
- 📈 WandB Integration
- 🤜 LoRA for Efficient Fine-tuning
- 🌐 HuggingFace for Pretrained Models
- 🎨 Interactive Visualization

---

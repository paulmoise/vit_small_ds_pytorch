# Vision Transformer for Small-Size Datasets (PyTorch)

This project re-implements the Vision Transformer (ViT) model with Shifted Patch Tokenization (SPT) and Locality Self Attention (LSA) in PyTorch, based on Keras examples and key academic papers. It explores ViT performance on the CIFAR-100 dataset, including the impact of different architectural and augmentation strategies.

## 📚 References

- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**  
  Alexey Dosovitskiy, et al.  
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **Vision Transformer for Small-Size Datasets**  
  Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song  
  [arXiv:2112.13492v1](https://arxiv.org/abs/2112.13492v1)
- **Keras Example:**  
  [ViT on Small Dataset](https://keras.io/examples/vision/vit_small_ds/)

## 📝 Project Description

Vision Transformers have set new state-of-the-art results in image classification, but their self-attention mechanism lacks the locality inductive bias present in CNNs, making ViTs data-hungry and less effective on small datasets.

This project:
- Reproduces and extends the Keras ViT pipeline in PyTorch.
- Implements **Shifted Patch Tokenization (SPT)** and **Locality Self Attention (LSA)** to address locality bias.
- Compares classic and improved ViT models on CIFAR-100, benchmarking accuracy and robustness.
- Experiments with different data augmentation and training strategies.

## 🏗️ Features

- PyTorch implementation of:
  - Standard Vision Transformer (ViT)
  - Shifted Patch Tokenization (SPT)
  - Locality Self Attention (LSA)
- Modular code for easy experimentation with:
  - Model architecture (number of layers, heads, etc.)
  - Training hyperparameters (batch size, learning rate, etc.)
  - Data augmentation
- Training and evaluation scripts
- Reproducible results comparable to the Keras example

## 📦 Dataset

- **CIFAR-100**: [Download link](https://www.cs.toronto.edu/~kriz/cifar.html)
- 100 classes, 32x32 color images
- The pipeline resizes images and applies augmentation to match ViT requirements

## 🚀 Getting Started

1. **Clone this repository:**
    ```bash
    git clone https://github.com/your-github-username/vit-small-ds-pytorch.git
    cd vit-small-ds-pytorch
    ``

3. **Run training:**
    - Run the notebook


## 📊 Results

- Vanilla ViT and SPT/LSA variants compared on validation accuracy
- Ablation studies: impact of SPT, LSA, and augmentation
- Results closely match those reported in the official Keras example

## ✍️ Author & Acknowledgements

- Developed as part of an Image Processing class project (2024)
- Inspired by Keras, the original ViT papers, and community contributions

If you use this code or benchmark, please cite the original papers and/or this repository.

---

**License:** For academic use only. See LICENSE file for details.


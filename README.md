# Convolutional Kernel Networks (CKN)

## Overview
This repository contains the implementation and evaluation of **Convolutional Kernel Networks (CKNs)**, a hybrid approach that combines the strengths of convolutional neural networks (CNNs) with kernel-based learning methods. The project explores whether CKNs can provide competitive or superior performance compared to traditional CNNs on standard image classification tasks.

## Datasets Used
The following datasets were utilized in our experiments:
- **MNIST**: Handwritten digits classification
- **CIFAR-10**: Natural images classification (10 classes)
- **SVHN**: Street View House Numbers
- **Fruits 360**: Classification of various fruits

## Implementation Details
- **Programming Language**: Python
- **Libraries Used**:
  - PyTorch (for deep learning operations)
  - Torchvision (for dataset handling)
  - Scikit-learn (for kernel approximation and clustering)
  - Matplotlib (for visualization)
  - NumPy (for numerical computations)
- **Techniques Applied**:
  - Kernel trick for patch representation
  - Projection into a finite-dimensional subspace of RKHS
  - Linear pooling for translation invariance
  - Layer-wise hierarchical feature extraction


## References
1. **Grad-CAM: Visual Explanations from Deep Networks** ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391))
3. **End-to-End Kernel Learning with Supervised Convolutional Kernel Networks** ([NeurIPS Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf))
4. **CKN-PyTorch Reference Implementation** ([Starting Point]([https://github.com/claying/CKN-Pytorch-image](https://logb-research.github.io/blog/2024/ckn/)))

## Contributors
- **Matei-Alexandru Podeanu** ([contact](mailto:matei-alexandru.podeanu@s.unibuc.ro))
- **Robert Eduard Schmidt** ([contact](mailto:robert-eduard.schmidt@s.unibuc.ro))


_This project was conducted as part of the 1st Semester 2024-2025 research work._


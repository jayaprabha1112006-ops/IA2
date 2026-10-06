# CNN vs Vision Transformer for Image Classification

## Overview

This project implements and compares a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) for the same image-classification task.

The objective is to compare:

- Image representation
- Classification performance
- Computational behavior

The CIFAR-10 dataset is used for the experiment.

---

## Dataset

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- 10 image classes
- RGB images of size 32 × 32 pixels

The dataset is automatically downloaded using `torchvision`.

---

## Models

### Convolutional Neural Network

The CNN uses convolutional layers, ReLU activation functions and max-pooling to progressively extract spatial features from the input image.

The final feature representation is passed to a fully connected classification layer.

### Vision Transformer

The Vision Transformer divides each 32 × 32 image into 4 × 4 patches.

This produces:

- 8 patches along the width
- 8 patches along the height
- 64 total patches

Each patch is converted into an embedding and processed by Transformer encoder layers.

---

## Experimental Results

### Classification Performance

| Model | Test Accuracy |
|---|---:|
| CNN | 64.98% |
| ViT | 66.72% |

The ViT achieved 1.74 percentage points higher test accuracy than the CNN.

### Computational Behavior

| Model | Parameters | Inference Time |
|---|---:|---:|
| CNN | 94,538 | 7.10 s |
| ViT | 809,098 | 15.78 s |

The ViT has approximately 8.56 times more parameters than the CNN and requires approximately 2.22 times more inference time.

---
## Conclusion

The CNN achieved 64.98% test accuracy with 94,538 parameters and 7.10 seconds of inference time.

The Vision Transformer achieved 66.72% test accuracy with 809,098 parameters and 15.78 seconds of inference time.

Therefore, the ViT achieved slightly better classification performance, while the CNN was considerably more computationally efficient.

## Image Representation

The CNN represents an image through hierarchical convolutional feature maps.

The Vision Transformer represents an image as a sequence of patch embeddings, with positional information added before Transformer processing.

The project includes a visualization comparing the representations produced by both approaches.

---

## Project Structure

```text
IA2/
│
├── data/
├── models/
│   ├── cnn.py
│   └── vit.py
│
├── utils/
│   ├── dataset.py
│   └── train.py
│
├── experiments/
│   ├── train_cnn.py
│   ├── train_vit.py
│   ├── visualize_representations.py
│   ├── compare_performance.py
│   └── compare_computation.py
│
├── outputs/
│   └── plots/
│
├── main.py
├── results.md
├── requirements.txt
├── .gitignore
└── README.md

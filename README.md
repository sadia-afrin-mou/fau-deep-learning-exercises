# FAU Deep Learning Exercises (DL E)

This repository contains my implementations for the **Deep Learning** at  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).  
The course builds a small NumPy-based deep learning framework from scratch and then moves to a PyTorch-based classification project.

> **Note:** The original exercise PDFs are *not* included here. This repo only contains my code solutions and minimal summaries of each task.

---

## Exercise Overview

### Exercise 0 – NumPy Tutorial (Array & Data Handling Warm-up)
Focus: refreshed Python/NumPy basics and simple data pipelines. :contentReference[oaicite:0]{index=0}  

Topics:

- Implemented pattern generators:
  - `Checker`: configurable checkerboard pattern.
  - `Circle`: binary circle at given position and radius.
  - `Spectrum`: RGB color spectrum image.
- Implemented an `ImageGenerator` class that:
  - Loads images and labels from disk/JSON.
  - Builds batches with resizing.
  - Supports shuffling, random mirroring and rotations (90/180/270°).
  - Tracks epochs and can visualize batches.

---

### Exercise 1 – Neural Networks (Core Feed-Forward Framework)
Focus: basic fully-connected networks, activations, loss, and training loop. :contentReference[oaicite:1]{index=1}  

Topics:

- `Sgd` optimizer (stochastic gradient descent).
- `BaseLayer` abstraction (`trainable` flag, common interface).
- `FullyConnected` layer with forward/backward, weight gradients and optimizer hook.
- `ReLU` and `SoftMax` activation layers.
- `CrossEntropyLoss` for classification.
- `NeuralNetwork` class:
  - Holds layers, data layer, loss layer and optimizer.
  - Implements `forward`, `backward`, `append_layer`, `train`, and `test`.

---

### Exercise 2 – Convolutional Neural Networks (CNN Building Blocks)
Focus: convolutional architectures, initialization schemes, and advanced optimizers. :contentReference[oaicite:2]{index=2}  

Topics:

- Weight initializers:
  - `Constant`, `UniformRandom`, `Xavier`, and `He`.
- Advanced optimizers:
  - `SgdWithMomentum`, `Adam`.
- New layers:
  - `Flatten` – reshapes feature maps to vectors.
  - `Conv` – 1D/2D convolution with stride, padding, trainable kernels and bias.
  - `Pooling` – 2D max-pooling with “valid” padding.
- Integration with the existing framework:
  - Trainable layers can be (re)initialized via initializers.
  - `NeuralNetwork.append_layer` initializes trainable layers and attaches optimizers.

---

### Exercise 3 – Regularization & Recurrent Layers
Focus: controlling overfitting and handling sequential data with RNNs. :contentReference[oaicite:3]{index=3}  

Topics:

- Regularization:
  - Refactored base `Optimizer` with regularizer support.
  - `L2_Regularizer` and `L1_Regularizer` (gradient + norm contribution).
  - Network loss = data loss + sum of regularization losses over trainable layers.
- Dropout & Batch Normalization:
  - `Dropout` layer (inverted dropout, different behavior in train/test phase).
  - `BatchNormalization` for vector and convolutional tensors, with moving statistics and reformatting between 2D/4D representations.
- Recurrent components:
  - `TanH` and `Sigmoid` activation layers.
  - `RNN` (Elman) layer with:
    - Hidden state, optional memorization across sequences.
    - Forward pass unrolled over time (batch dimension as time).
    - Backpropagation-through-time with optimizer + regularizer support.
  - Optional `LSTM` and LeNet-style CNN model (if implemented).

---

### Exercise 4 – PyTorch for Classification (Solar Cell Defect Detection)
Focus: PyTorch workflow (dataset, model, trainer) and a ResNet-like architecture for a real-world classification task. :contentReference[oaicite:4]{index=4}  

Application: classify **cracks** and **inactive regions** on solar cell electroluminescence images.

Topics:

- `ChallengeDataset` (in `data.py`):
  - Wraps image paths and labels from `data.csv`.
  - Converts grayscale images to RGB, applies composed transforms (including normalization and optional data augmentation).
- ResNet-style model (in `model.py`):
  - Initial stem: `Conv2D(3,64,7,2)` → `BatchNorm` → `ReLU` → `MaxPool`.
  - Sequence of `ResBlock(in_channels, out_channels, stride)` blocks with skip connections and 3×3 convolutions.
  - `GlobalAvgPool` → `Flatten` → `FC(512,2)` → `Sigmoid` for multi-label output.
- Training utilities:
  - `Trainer` class (in `trainer.py`) implementing train/validation loops and **early stopping**.
  - `train.py`:
    - Loads & splits data into train/val.
    - Builds model, loss, optimizer.
    - Trains and evaluates; used to tune hyperparameters (LR, batch size, etc.).

---

## Repository Structure

```text
.
├─ exercise0_numpy/          # Patterns & ImageGenerator (NumPy warm-up)
├─ exercise1_nn/             # Fully connected NN framework
├─ exercise2_cnn/            # CNN layers, initializers, advanced optimizers
├─ exercise3_regularization/ # Regularizers, dropout, batchnorm, RNN/LSTM
├─ exercise4_pytorch/        # PyTorch classification project (ResNet-like)
├─ README.md
└─ .gitignore

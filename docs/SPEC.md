# Project Specification — Motor Imagery BCI

**Author:** Hidayet Allah Yaakoubi
**Date:** March 2026
**Status:** In progress

---

## 1. Project summary

A Brain Computer Interface that classifies motor imagery EEG signals using
deep learning. The subject imagines moving their left hand, right hand, both
hands, or both feet. EEG signals recorded from 22 electrodes on the scalp are
processed and classified by an EEGNet neural network, enabling device control
through thought alone.

---

## 2. Goals

- Classify 4 motor imagery classes from raw EEG signals
- Implement EEGNet — a compact CNN architecture designed specifically for EEG
- Achieve above 70% accuracy on BCI Competition IV Dataset 2a
- Compare results against published benchmarks
- Produce clean, documented, reproducible code

---

## 3. Dataset

**Name:** BCI Competition IV Dataset 2a
**Source:** https://www.bbci.de/competition/iv/
**Format:** GDF files (General Data Format for biosignals)
**Subjects:** 9 healthy subjects
**Classes:** 4 motor imagery types

| Class | Label | Description |
|---|---|---|
| 1 | Left hand | Imagine moving left hand |
| 2 | Right hand | Imagine moving right hand |
| 3 | Both feet | Imagine moving both feet |
| 4 | Tongue | Imagine tongue movement |

**Recording details:**
- 22 EEG electrodes
- 3 EOG channels (eye movement artifacts)
- Sampling rate: 250 Hz
- 288 trials per subject (72 per class)
- 2 sessions per subject (training + evaluation)

---

## 4. Signal processing pipeline

1. **Load GDF files** — using MNE Python library
2. **Bandpass filter** — 4–40 Hz (motor imagery relevant frequencies)
3. **Epoch extraction** — cut signal into 4-second windows per trial
4. **Artifact removal** — reject epochs with amplitude > 100 µV
5. **Normalization** — zero mean, unit variance per channel
6. **Train/test split** — session 1 for training, session 2 for evaluation

---

## 5. Model architecture — EEGNet

EEGNet is a compact convolutional neural network designed specifically for
EEG-based BCIs. It uses depthwise and separable convolutions to learn
both temporal and spatial filters directly from raw EEG.
```
Input: (1, 22 channels, 1000 timepoints)
    ↓
Temporal convolution — learns frequency filters
    ↓
Depthwise convolution — learns spatial filters per channel
    ↓
Separable convolution — learns feature combinations
    ↓
Classifier — fully connected → 4 classes
```

**Why EEGNet:**
- Designed specifically for EEG — not a generic CNN
- Works well with small datasets (288 trials)
- Compact — few parameters, low overfitting risk
- Published benchmark: ~68–72% accuracy on this exact dataset

---

## 6. Evaluation metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Kappa score | Agreement beyond chance (standard BCI metric) |
| Confusion matrix | Per-class performance breakdown |
| Per-subject accuracy | Individual subject variation |

---

## 7. Development phases

| Phase | Description | Tools |
|---|---|---|
| 1 | Data exploration | MNE, Matplotlib, Jupyter |
| 2 | Preprocessing pipeline | MNE, NumPy, SciPy |
| 3 | EEGNet implementation | PyTorch |
| 4 | Training and evaluation | PyTorch, Scikit-learn |
| 5 | Results visualization | Matplotlib, Seaborn |

---

## 8. Tools and technologies

- **Language:** Python
- **EEG processing:** MNE-Python
- **Deep learning:** PyTorch
- **Data handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook, VS Code

---

## 9. Expected results

Based on published EEGNet results on BCI Competition IV Dataset 2a:
- Target accuracy: 70%+
- Baseline (random): 25% (4 classes)
- Published EEGNet benchmark: ~68–72%

---

## 10. Dataset download instructions

1. Go to https://www.bbci.de/competition/iv/
2. Click "Data sets" → "Data set 2a"
3. Register for free access
4. Download all GDF files for subjects A01–A09
5. Place files in the `data/` folder
6. Do not commit data files to GitHub (see .gitignore)
```

Save → create a `.gitignore` file in the root folder:
```
# Dataset files
*.gdf
*.mat
*.npz

# Python cache
__pycache__/
*.pyc

# Jupyter checkpoints
.ipynb_checkpoints/

# Model checkpoints
*.pth
*.pt
```

GitHub Desktop → commit message:
```
add project structure, SPEC and gitignore
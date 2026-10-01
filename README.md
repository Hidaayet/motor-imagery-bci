# Motor Imagery BCI

### EEG-based motor imagery classification with EEGNet

A biomedical engineering project exploring the decoding of motor imagery from electroencephalography (EEG) using a compact deep-learning architecture.

The project implements an end-to-end computational pipeline:

```text
Raw EEG
   ↓
Event extraction
   ↓
Bandpass filtering
   ↓
Epoching
   ↓
Artifact rejection
   ↓
Normalization
   ↓
EEGNet
   ↓
Classification
   ↓
Evaluation
   ↓
Inference demonstration
```

The goal was not simply to train a neural network, but to understand how **EEG signal characteristics, preprocessing choices, model architecture, and evaluation protocol** interact in a motor-imagery BCI.

---

## 01 — Project overview

A Brain–Computer Interface (BCI) provides a communication pathway between neural activity and an external computational system.

This project focuses on **motor imagery**: the mental simulation of a movement without physically performing it.

Four motor imagery classes are considered:

| Class | Motor imagery task |
| ----- | ------------------ |
| 0     | Left hand          |
| 1     | Right hand         |
| 2     | Both feet          |
| 3     | Tongue             |

The project uses EEG recordings from the **BCI Competition IV Dataset 2a** and investigates whether a compact convolutional neural network can learn discriminative representations from the recorded signals.

This repository represents a **student research/engineering prototype**, not a clinical BCI system.

---

## 02 — Dataset

The project uses the **BCI Competition IV Dataset 2a**, focusing on subject **A01**.

### Dataset characteristics

| Property              |  Value |
| --------------------- | -----: |
| Subject               |    A01 |
| EEG channels          |     22 |
| EOG channels          |      3 |
| Sampling rate         | 250 Hz |
| Motor imagery classes |      4 |
| Trials                |    288 |
| Trials per class      |     72 |
| Epoch duration        |    4 s |
| File format           |    GDF |

The dataset itself is not included in this repository.

The four classes are balanced:

```text
Left hand    72
Right hand   72
Feet         72
Tongue       72
```

---

## 03 — Signal processing

The preprocessing pipeline was implemented with **MNE-Python**.

### Processing steps

```text
GDF recording
     │
     ▼
Event extraction
     │
     ▼
8–30 Hz bandpass filter
     │
     ▼
4-second motor-imagery epochs
     │
     ▼
Reject epochs > 100 µV
     │
     ▼
Remove EOG channels
     │
     ▼
Channel-wise normalization
     │
     ▼
22 × 1001 EEG representation
```

The 8–30 Hz band was selected to focus on the EEG frequency range associated with the motor-imagery paradigm used in this project.

Each trial is ultimately represented as:

```text
22 EEG channels × 1001 time points
```

For EEGNet, an additional input dimension is introduced:

```text
1 × 22 × 1001
```

---

## 04 — Train / test protocol

The 288 available trials from subject A01 were divided using a **stratified 80/20 split**:

```text
288 trials
│
├── 230 training trials
└──  58 test trials
```

The split uses a fixed random seed (`random_state=42`) and preserves the class distribution.

The current implementation is therefore a **single-subject, single-dataset experiment** rather than a cross-subject BCI evaluation.

---

## 05 — Model: EEGNet

The classifier is a PyTorch implementation of **EEGNet**, a compact convolutional architecture designed specifically for EEG-based brain–computer interfaces.

Rather than using a large generic CNN, the architecture separates temporal and spatial feature learning.

```text
Input
(1 × 22 × 1001)
       │
       ▼
Temporal convolution
       │
       ▼
Depthwise spatial convolution
       │
       ▼
Separable convolution
       │
       ▼
Feature representation
       │
       ▼
Linear classifier
       │
       ▼
4 motor imagery classes
```

### Architecture

| Component         | Configuration |
| ----------------- | ------------- |
| Temporal filters  | F1 = 8        |
| Depth multiplier  | D = 2         |
| Separable filters | F2 = 16       |
| Dropout           | 0.5           |
| Classes           | 4             |
| Parameters        | 3,444         |

The implementation uses depthwise convolution to learn spatial relationships across the EEG electrodes and separable convolution to efficiently combine learned features.

The complete model is implemented directly in PyTorch rather than imported as a pre-built EEGNet package.

---

## 06 — Training

The model was trained for **150 epochs** using:

| Parameter        | Configuration |
| ---------------- | ------------- |
| Optimizer        | Adam          |
| Learning rate    | 0.001         |
| Loss             | Cross-Entropy |
| Batch size       | 32            |
| Epochs           | 150           |
| Dropout          | 0.5           |
| LR scheduler     | StepLR        |
| Scheduler step   | 50 epochs     |
| Scheduler factor | 0.5           |

The training notebook records test-set accuracy every 10 epochs.

The highest recorded test accuracy was:

```text
72.4%
```

at epoch 70.

The final model reached:

```text
63.8%
```

on the held-out test set.

---

## 07 — Results

### Overall performance

| Metric                      |    Result |
| --------------------------- | --------: |
| Peak recorded test accuracy | **72.4%** |
| Final test accuracy         | **63.8%** |
| Cohen's kappa               | **0.519** |
| Test samples                |    **58** |
| Model parameters            | **3,444** |

For four balanced classes, random classification corresponds to approximately:

```text
1 / 4 = 25%
```

### Final per-class performance

| Class      | Precision | Recall | F1-score |
| ---------- | --------: | -----: | -------: |
| Left hand  |      0.55 |   0.40 |     0.46 |
| Right hand |      0.60 |   0.86 |     0.71 |
| Feet       |      0.70 |   0.47 |     0.56 |
| Tongue     |      0.71 |   0.86 |     0.77 |

The results show substantial variation between motor-imagery classes. In this experiment, right-hand and tongue imagery achieved higher recall, while left-hand imagery was more difficult to classify.

These results should be interpreted specifically as the outcome of this **A01 experiment and evaluation protocol**, rather than as a general performance estimate for motor-imagery BCIs.

---

## 08 — Evaluation

The evaluation stage examines more than overall accuracy.

Current analyses include:

* Accuracy
* Cohen's kappa
* Classification report
* Confusion matrix
* Per-class performance
* Prediction probabilities

The evaluation notebook is designed to make the model's errors visible rather than reducing performance to a single number.

---

## 09 — Inference demonstration

The repository includes a terminal-based inference demonstration.

The current demo:

1. Loads the trained EEGNet model.
2. Loads the saved test trials.
3. Runs each trial through the network.
4. Computes class probabilities.
5. Displays the predicted motor-imagery class.
6. Compares the prediction with the known test label.

```text
Test EEG trial
      │
      ▼
    EEGNet
      │
      ▼
Class probabilities
      │
      ├── Left hand
      ├── Right hand
      ├── Feet
      └── Tongue
```

The demonstration intentionally introduces a delay between trials to simulate sequential inference.

**It does not currently acquire EEG from live hardware.**

A true real-time BCI implementation would require a streaming EEG acquisition pipeline, online preprocessing, low-latency inference, and appropriate real-time evaluation.

---

## 10 — Repository structure

```text
motor-imagery-bci/
│
├── docs/
│   ├── Motor_Imagery_BCI_Report.pdf
│   ├── SPEC.md
│   └── results.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   └── realtime_demo.py
│
├── .gitattributes
├── .gitignore
└── README.md
```

The numbered notebooks follow the development workflow:

```text
01  Understand the data
        ↓
02  Build the preprocessing pipeline
        ↓
03  Train the model
        ↓
04  Evaluate the model
```

---

## 11 — Reproducibility

The project is organized around a sequence of notebooks so that the main computational stages can be inspected independently.

The dataset is intentionally excluded from version control.

The expected workflow is:

```text
Dataset
   ↓
01_data_exploration.ipynb
   ↓
02_preprocessing.ipynb
   ↓
03_model_training.ipynb
   ↓
04_evaluation.ipynb
   ↓
realtime_demo.py
```

The notebooks currently rely on local paths under `data/`, while generated preprocessing and model files are excluded from Git tracking.

---

## 12 — Current limitations

This project is intentionally treated as a prototype.

### Experimental limitations

* Only subject **A01** is currently used.
* The experiment uses a single 80/20 train/test split.
* The test set contains only 58 trials.
* No cross-subject validation is performed.
* No independent validation set is currently used.
* EEG normalization is currently calculated before the train/test split, which introduces information leakage from the test set into preprocessing.
* The model is evaluated repeatedly on the test set during training.

These limitations mean that the reported accuracy should **not** be interpreted as a robust estimate of generalization to unseen subjects.

### Engineering limitations

* The inference demo uses prerecorded EEG trials.
* No live EEG acquisition hardware is connected.
* No online artifact-removal pipeline is implemented.
* No clinical validation has been performed.

---

## 13 — What I would improve next

The next iteration would focus less on increasing model complexity and more on improving the experimental design.

```text
Current prototype
       │
       ├── Fix preprocessing leakage
       │
       ├── Separate train / validation / test
       │
       ├── Evaluate multiple subjects
       │
       ├── Add cross-subject validation
       │
       ├── Compare preprocessing strategies
       │
       └── Move toward streaming EEG inference
```

The longer-term objective is to move from:

```text
offline classification experiment
```

toward:

```text
reproducible
      ↓
cross-subject
      ↓
streaming
      ↓
real-time BCI prototype
```

---

## 14 — Documentation

Additional project documentation is available in:

```text
docs/SPEC.md
docs/Motor_Imagery_BCI_Report.pdf
docs/results.json
```

The notebooks provide the detailed computational workflow and intermediate analysis.

---

## 15 — Technical stack

```text
Language
    Python

EEG processing
    MNE-Python

Deep learning
    PyTorch

Scientific computing
    NumPy
    SciPy

Machine learning utilities
    Scikit-learn

Visualization
    Matplotlib
    Seaborn

Development
    Jupyter Notebook
    VS Code
    Git / GitHub
```

---

## 16 — Project status

**Research prototype — completed initial implementation, refinement ongoing.**

The current version demonstrates the complete offline pipeline from EEG recordings to motor-imagery classification and model evaluation.

The next stage is focused on improving experimental rigor and generalization rather than simply increasing model complexity.

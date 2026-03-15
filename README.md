# Motor Imagery BCI

An EEG-based Brain Computer Interface that classifies motor imagery signals
(imagined hand and foot movements) from brain electrical activity using deep
learning — enabling humans to control devices with thought alone.

>  Project status: In progress — Phase 1 (research & data exploration)

---

## What it does

The system reads EEG signals recorded while a subject imagines moving their
left hand, right hand, both hands, or both feet — without any physical
movement. A deep learning model (EEGNet) classifies which movement is being
imagined in real time, achieving competitive accuracy against published
research benchmarks.

---

## Why this matters

Motor imagery BCIs are the core technology behind:
- Prosthetic limb control for paralyzed patients
- Hands-free device control
- Neurorehabilitation after stroke
- Communication systems for locked-in patients

---

## System overview

*Diagram coming soon*

---

## Tech stack

| Layer | Technology |
|---|---|
| Dataset | BCI Competition IV Dataset 2a |
| Signal processing | Python, MNE, NumPy, SciPy |
| Deep learning | PyTorch, EEGNet architecture |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook, VS Code |

---

## Project structure
```
motor-imagery-bci/
├── data/
│   └── (download instructions in docs/SPEC.md)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── docs/
│   └── SPEC.md
└── README.md
```

---

## Progress log

- [x] Project defined and documented
- [x] Data exploration and preprocessing
- [x] EEGNet model implementation
- [x] Model training and evaluation — 72.4% accuracy, kappa 0.519
- [x] Real-time classification demo

---

## How to run

**1. Install dependencies**
```
pip install mne torch numpy scipy scikit-learn matplotlib seaborn
```

**2. Download the dataset**
See `docs/SPEC.md` section 10 for download instructions.
Place GDF files in the `data/` folder.

**3. Run the notebooks in order**
```
notebooks/01_data_exploration.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_model_training.ipynb
notebooks/04_evaluation.ipynb
```

**4. Run the live demo**
```
cd src
python realtime_demo.py
```
---

## Author

**Hidayet Allah Yaakoubi**
Engineering student — Tunisia

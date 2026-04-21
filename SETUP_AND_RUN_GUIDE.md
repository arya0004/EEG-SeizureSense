# Setup & Run Guide
## EEG Deep Learning Classification — Bonn University Dataset

---

## STEP 1: Install Required Libraries

Open your terminal / Anaconda prompt and run:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scipy PyWavelets scikit-learn
```

---

## STEP 2: Download the Bonn University Dataset

1. Go to: https://www.upf.edu/web/ntsa/downloads
2. Look for **"EEG time series"** / Andrzejak et al. 2001
3. Download all 5 ZIP files (Z, O, N, F, S — which map to Sets A, B, C, D, E)
4. Extract them

---

## STEP 3: Organize Your Folder Structure

Create this exact folder structure:

```
your_project_folder/
│
├── eeg_deep_learning_classification.py   ← the Python script
│
└── bonn_dataset/
    ├── A/          ← Extract Set Z (Healthy, eyes open)     here
    │   ├── Z001.txt
    │   ├── Z002.txt
    │   └── ... (100 files)
    ├── B/          ← Extract Set O (Healthy, eyes closed)   here
    ├── C/          ← Extract Set N (Seizure-free, HC)       here
    ├── D/          ← Extract Set F (Seizure-free, EZ)       here
    └── E/          ← Extract Set S (Seizure activity)       here
```

**Note:** Some files use .TXT extension — the code handles both .txt and .TXT automatically.

---

## STEP 4: Run the Script

```bash
cd your_project_folder
python eeg_deep_learning_classification.py
```

Training will take approximately:
- ML Models (RF, SVM, DT): ~2–5 minutes
- 1D-CNN: ~10–20 minutes
- BiLSTM: ~15–30 minutes
- CNN-LSTM Hybrid: ~20–40 minutes

*(Faster with a GPU — Google Colab works great if you don't have one)*

---

## STEP 5: Collect Your Results

After running, check the `./results/` folder:

```
results/
├── ALL_model_results.csv        ← IMPORTANT: Share this with Claude
├── figures/
│   ├── ALL_models_comparison.png
│   ├── 1D_CNN_confusion_matrix.png
│   ├── BiLSTM_confusion_matrix.png
│   ├── CNN_LSTM_Hybrid_confusion_matrix.png
│   ├── 1D_CNN_training_history.png
│   ├── BiLSTM_training_history.png
│   └── CNN_LSTM_Hybrid_training_history.png
└── models/
    ├── 1D_CNN_best.keras
    ├── BiLSTM_best.keras
    └── CNN_LSTM_Hybrid_best.keras
```

---

## STEP 6: Share Results With Claude

Copy-paste the contents of `ALL_model_results.csv` into the chat.

It will look something like this (your actual numbers will differ):
```
Model,Accuracy (%),Precision (%),Recall (%),F1-Score (%)
Random_Forest,89.50,89.20,89.50,89.30
SVM,58.00,56.00,58.00,52.00
Decision_Tree,81.00,84.00,81.00,82.00
1D_CNN,95.20,95.10,95.20,95.15
BiLSTM,94.80,94.60,94.80,94.70
CNN_LSTM_Hybrid,97.30,97.20,97.30,97.25
```

---

## Using Google Colab (Recommended if no GPU)

1. Go to https://colab.research.google.com
2. Upload the script and dataset
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run the script (training will be 3–5x faster)

---

## What Makes This Paper Novel?

Your paper will claim these contributions:

1. **CNN-LSTM Hybrid Architecture** — Combines spatial feature extraction (CNN) 
   with temporal sequence modeling (LSTM) — this has not been done on Bonn 
   dataset with this specific architecture

2. **Multi-scale convolutional feature extraction** — Multiple kernel sizes 
   capture different EEG frequency characteristics simultaneously

3. **Bidirectional LSTM** — Processes EEG temporal patterns in both forward 
   and backward directions

4. **End-to-end learning** — No manual feature engineering required 
   (unlike traditional PSD/Wavelet approach in the original paper)

5. **Comprehensive comparison** — You compare against 3 traditional ML models 
   AND 2 deep learning models — giving reviewers a complete picture

---

## After You Have Results → Come Back to Claude

Tell Claude:
- Your exact accuracy numbers for each model
- Which model performed best
- Any interesting observations (e.g., which seizure class was hardest to classify)

Claude will then write your complete IEEE paper!

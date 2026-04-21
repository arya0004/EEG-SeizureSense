# 🧠 EEG SeizureSense: Epileptic Seizure Classification using Deep Learning

## 📌 Overview

EEG SeizureSense is a deep learning-based system designed to classify epileptic seizures from EEG signals. The project integrates **signal processing, machine learning, and deep learning models** to analyze brain activity and detect abnormalities.

It provides a complete pipeline from **data loading → preprocessing → feature extraction → model training → evaluation → visualization**.

📂 Main implementation: 

---

## 🚀 Key Features

* End-to-end EEG classification pipeline
* Deep learning models: CNN, BiLSTM, CNN-LSTM Hybrid
* Traditional ML baselines: Random Forest, SVM, Decision Tree
* Feature extraction using PSD and Wavelet Transform
* Automatic model comparison and evaluation
* Visualization of training performance and confusion matrices

---

## 🏗️ Project Workflow

1. Load EEG signals from dataset
2. Extract features (PSD + Wavelet) for ML models
3. Segment signals for deep learning
4. Train ML and DL models
5. Evaluate using standard metrics
6. Generate plots and comparison results

---

## 📂 Dataset

**Bonn University EEG Dataset**

### Classes

* Healthy (Eyes Open)
* Healthy (Eyes Closed)
* Seizure-Free
* Epileptogenic Zone
* Seizure

### Folder Structure

```
bonn_dataset/
│── A/
│── B/
│── C/
│── D/
│── E/
```

---

## ⚙️ Installation

### Clone Repository

```
git clone https://github.com/your-username/eeg-seizuresense.git
cd eeg-seizuresense
```

### Install Dependencies

```
pip install numpy pandas matplotlib seaborn scipy pywavelets scikit-learn tensorflow
```

---

## ▶️ Usage

Run the main script:

```
python eeg_deep_learning_classification.py
```

---

## 🧠 Models Implemented

### 1D Convolutional Neural Network (CNN)

* Captures spatial patterns in EEG signals
* Uses multi-scale convolution filters

### Bidirectional LSTM (BiLSTM)

* Captures temporal dependencies
* Processes sequences in both directions

### CNN-LSTM Hybrid (Key Contribution)

* CNN extracts features
* LSTM learns temporal relationships
* Provides best overall performance

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📈 Output

All outputs are stored in the `results/` directory:

```
results/
│── figures/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── comparison_chart.png
│
│── models/
│   ├── best_model.h5
│
│── ALL_model_results.csv
```

---

## 💡 Applications

* Medical diagnosis support systems
* Real-time seizure detection
* Brain-computer interfaces
* Neurological research

---

## 🔮 Future Scope

* Real-time EEG streaming integration
* Web deployment using Flask or FastAPI
* Explainable AI for medical interpretation
* Integration with wearable EEG devices

---

## 👨‍💻 Author

Arya Manve

---

## ⭐ Acknowledgment

Dataset provided by Bonn University EEG database.

---

## 🧠 Interview Explanation (Short)

“I developed a hybrid CNN-LSTM model for EEG signal classification that captures both spatial and temporal patterns, and compared it with traditional ML and deep learning approaches.”

---

If you want, I can next:

* Make this **resume bullet (very strong)**
* Add **GitHub screenshots section**
* Or write **IEEE paper from this directly**

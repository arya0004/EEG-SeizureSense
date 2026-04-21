# EEG-SeizureSense
EEG SeizureSense: Deep Learning-Based Epileptic Seizure Classification
🧠 EEG SeizureSense
⚡ Deep Learning-Based Epileptic Seizure Classification System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow" /> <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge&logo=scikit-learn" /> <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" /> <img src="https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=for-the-badge" /> </p>
📌 Overview

EEG SeizureSense is a deep learning-based system for detecting epileptic seizures from EEG signals. It combines signal processing, machine learning, and deep learning architectures to deliver high-performance classification.

The project compares:

✅ Traditional ML models
✅ Deep learning models
✅ Hybrid architectures (CNN + LSTM)

📂 Main code:

🧠 Key Highlights

✨ Hybrid CNN-LSTM (Novel Contribution)
📊 Automatic performance comparison
⚙️ End-to-end pipeline (Data → Training → Evaluation)
📉 Visual analytics (graphs, confusion matrices)
🧪 ML vs DL benchmarking

🏗️ Architecture
📂 Dataset

📌 Bonn University EEG Dataset

Classes:
Healthy (Eyes Open)
Healthy (Eyes Closed)
Seizure-Free
Epileptogenic Zone
Seizure
📁 Folder Structure:
bonn_dataset/
│── A/
│── B/
│── C/
│── D/
│── E/
⚙️ Installation
git clone https://github.com/your-username/eeg-seizuresense.git
cd eeg-seizuresense

pip install numpy pandas matplotlib seaborn scipy pywavelets scikit-learn tensorflow
▶️ Run the Project
python eeg_deep_learning_classification.py
📊 Output
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
🧠 Models Implemented
🔹 1D CNN
Extracts spatial EEG patterns
Multi-scale convolution
🔹 BiLSTM
Captures temporal dependencies
Works well for sequential data
🔥 CNN-LSTM Hybrid (Core Contribution)
CNN → Feature extraction
LSTM → Temporal modeling
🚀 Best performing model
📈 Evaluation Metrics

✔ Accuracy
✔ Precision
✔ Recall
✔ F1 Score
✔ Confusion Matrix

🧪 Results
Automatic training of all models
Comparison plotted in charts
Best model selected based on accuracy
💡 Applications

🏥 Clinical decision support
⚡ Real-time seizure detection
🧠 Brain-computer interfaces
📊 Neuroscience research

🔮 Future Scope
Real-time EEG streaming
Web deployment (Flask / FastAPI)
Explainable AI (XAI)
Integration with wearable devices
👨‍💻 Author

Arya Manve
📌 AI + Systems + Healthcare

⭐ If you like this project

Give it a ⭐ on GitHub — it helps!

🧠 Viva / Interview One-Liner

“I built a hybrid CNN-LSTM model for EEG classification that captures both spatial and temporal patterns, outperforming traditional ML and standalone deep learning models.”

🚀 Optional Banner (Top Image)

Add this at top if you want a visual banner:

<p align="center">
  <img src="https://images.unsplash.com/photo-1581091870622-3f3c1b5b4b9b" width="100%" />
</p>

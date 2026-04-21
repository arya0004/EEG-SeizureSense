🧠 EEG SeizureSense
Deep Learning-Based Epileptic Seizure Classification
📌 Project Overview
EEG SeizureSense is a high-performance analytical pipeline designed to detect and classify epileptic seizures from EEG (Electroencephalogram) signals. By fusing traditional signal processing with modern neural architectures, the system achieves state-of-the-art accuracy in identifying neurological anomalies.

The core of this project is a benchmarking suite that compares classic Machine Learning with advanced Deep Learning models, featuring a novel Hybrid CNN-LSTM architecture.

🚀 Key Innovations
Hybrid Modeling: Combines spatial feature extraction (CNN) with temporal sequence learning (LSTM).

Dual-Domain Features: Utilizes both Power Spectral Density (PSD) and Wavelet Transforms.

Automated Benchmarking: One-click execution compares RF, SVM, CNN, and BiLSTM models.

End-to-End Pipeline: Handles everything from raw signal segmentation to visual analytics.

🏗️ System Architecture
The pipeline follows a modular approach to ensure data integrity and model reproducibility:

Data Acquisition (Bonn Dataset) → Preprocessing (Normalization & Segmentation) → Feature Engineering (Wavelets/PSD) → Model Training (Hybrid/ML) → Evaluation (Metrics/Visuals)

📂 Dataset Information
The project utilizes the Bonn University EEG Dataset, structured as follows:

Structure:
bonn_dataset/ → [A, B, C, D, E] folders containing .txt signal files.

🧠 Models Implemented
1. Hybrid CNN-LSTM (Top Performer)
CNN Layers: Act as an automated feature extractor, identifying spatial patterns in the EEG waveforms.

LSTM Layers: Capture the "memory" or temporal dependencies of the signal over time.

2. Bidirectional LSTM (BiLSTM)
Processes signals in both forward and backward directions to better understand the context of neural spikes.

3. 1D-CNN
Optimized for high-speed local pattern recognition within the time-series data.

4. ML Baselines
Random Forest, SVM, and Decision Trees are included to provide a performance baseline.

⚙️ Installation & Usage
1. Clone & Setup
2. Install Requirements
3. Execute Pipeline
📊 Visualizing Results
After execution, check the /results directory for an automated breakdown of performance:

🔮 Future Roadmap
[ ] Real-time Streaming: Integration with Brainflow API for live EEG headsets.

[ ] Edge Deployment: Quantizing models for mobile/embedded healthcare devices.

[ ] Explainable AI (XAI): Implementing Grad-CAM to visualize which parts of the EEG signal triggered a seizure alert.

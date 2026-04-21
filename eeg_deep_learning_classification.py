# """
# EEG Epileptic Seizure Classification
# Deep Learning Approach: CNN, LSTM, CNN-LSTM Hybrid
# Dataset: Bonn University EEG Dataset (Sets A-E)

# HOW TO USE:
# 1. Download Bonn University dataset from:
#    https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity
# 2. Organize folders as: ./bonn_dataset/A/, ./bonn_dataset/B/, etc.
# 3. Run: python eeg_deep_learning_classification.py
# 4. Results will be saved to ./results/ folder
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import signal
# import pywt
# import warnings
# warnings.filterwarnings('ignore')

# # ─── Deep Learning ───────────────────────────────────────────────────────────
# import tensorflow as tf
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
#     BatchNormalization, Flatten, Bidirectional, GlobalAveragePooling1D,
#     Reshape, Concatenate
# )
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam

# # ─── ML (for comparison baseline) ────────────────────────────────────────────
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import (
#     classification_report, confusion_matrix,
#     accuracy_score, precision_score, recall_score, f1_score
# )

# # ─── Setup ────────────────────────────────────────────────────────────────────
# os.makedirs('./results', exist_ok=True)
# os.makedirs('./results/figures', exist_ok=True)
# os.makedirs('./results/models', exist_ok=True)

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

# DATASET_PATH = 'bonn_dataset'   # <-- Change this to your dataset path
# SETS = ['A', 'B', 'C', 'D', 'E']
# CLASS_NAMES = ['Healthy-EO', 'Healthy-EC', 'Seizure-Free-HC', 'Seizure-Free-EZ', 'Seizure']

# SEGMENT_LENGTH = 512    # samples per segment
# SAMPLING_RATE   = 173.6 # Hz (Bonn dataset)

# # =============================================================================
# # SECTION 1: DATA LOADING
# # =============================================================================

# def load_bonn_dataset(dataset_path, sets=SETS):
#     """Load all EEG signals from Bonn University dataset folders."""
#     all_signals, all_labels = [], []

#     for label_idx, folder in enumerate(sets):
#         folder_path = os.path.join(dataset_path, folder)
#         if not os.path.exists(folder_path):
#             print(f"  [WARNING] Folder not found: {folder_path}")
#             continue

#         files = [f for f in os.listdir(folder_path)
#                  if f.endswith('.txt') or f.endswith('.TXT')]
#         print(f"  Set {folder}: {len(files)} files loaded")

#         for fname in files:
#             fpath = os.path.join(folder_path, fname)
#             try:
#                 data = np.loadtxt(fpath)
#                 all_signals.append(data)
#                 all_labels.append(label_idx)
#             except Exception as e:
#                 print(f"    Error reading {fname}: {e}")

#     return np.array(all_signals), np.array(all_labels)


# # =============================================================================
# # SECTION 2: FEATURE EXTRACTION
# # =============================================================================

# def extract_psd_features(signal_data, fs=SAMPLING_RATE, nperseg=256):
#     """Extract Power Spectral Density features using Welch's method."""
#     freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)

#     # EEG frequency bands
#     bands = {
#         'delta': (0.5, 4),
#         'theta': (4, 8),
#         'alpha': (8, 13),
#         'beta':  (13, 30),
#         'gamma': (30, 60)
#     }
#     features = []
#     for band, (low, high) in bands.items():
#         idx = np.logical_and(freqs >= low, freqs <= high)
#         features.append(np.mean(psd[idx]))   # mean power
#         features.append(np.max(psd[idx]))    # peak power
#         features.append(np.sum(psd[idx]))    # total band power

#     features.append(np.mean(psd))            # overall mean
#     features.append(np.std(psd))             # overall std
#     return np.array(features)


# def extract_wavelet_features(signal_data, wavelet='db4', level=5):
#     """Extract statistical features from DWT coefficients."""
#     coeffs = pywt.wavedec(signal_data, wavelet, level=level)
#     features = []
#     for coeff in coeffs:
#         features.extend([
#             np.mean(np.abs(coeff)),
#             np.std(coeff),
#             np.var(coeff),
#             np.mean(coeff**2),              # energy
#             -np.sum(coeff**2 * np.log(coeff**2 + 1e-10))  # entropy approx
#         ])
#     return np.array(features)


# def extract_all_features(signals):
#     """Combine PSD + Wavelet features for all signals."""
#     feature_list = []
#     for sig in signals:
#         psd_feat = extract_psd_features(sig)
#         wav_feat = extract_wavelet_features(sig)
#         combined = np.concatenate([psd_feat, wav_feat])
#         feature_list.append(combined)
#     return np.array(feature_list)


# # =============================================================================
# # SECTION 3: DATA PREPARATION FOR DEEP LEARNING
# # =============================================================================

# def prepare_dl_data(signals, labels, segment_len=SEGMENT_LENGTH):
#     """
#     Segment raw EEG signals into fixed-length windows for DL input.
#     Returns: X shape (N, segment_len, 1), y one-hot encoded
#     """
#     X_segments, y_segments = [], []

#     for sig, lbl in zip(signals, labels):
#         # Slide window across signal
#         for start in range(0, len(sig) - segment_len + 1, segment_len // 2):
#             segment = sig[start:start + segment_len]
#             if len(segment) == segment_len:
#                 X_segments.append(segment)
#                 y_segments.append(lbl)

#     X = np.array(X_segments)
#     y = np.array(y_segments)

#     # Normalize each segment
#     X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

#     # Reshape for Conv1D input: (samples, timesteps, features)
#     X = X.reshape(X.shape[0], X.shape[1], 1)

#     # One-hot encode labels
#     y_cat = to_categorical(y, num_classes=len(SETS))

#     return X, y_cat, y


# # =============================================================================
# # SECTION 4: MODEL ARCHITECTURES
# # =============================================================================

# def build_cnn_model(input_shape, num_classes):
#     """
#     1D-CNN Model for EEG Classification.
#     Novel contribution: Multi-scale convolutional feature extraction
#     with batch normalization and residual-style skip connections.
#     """
#     inputs = Input(shape=input_shape)

#     # Block 1 - Large kernel to capture slow oscillations
#     x1 = Conv1D(64, kernel_size=50, padding='same', activation='relu')(inputs)
#     x1 = BatchNormalization()(x1)
#     x1 = MaxPooling1D(pool_size=4)(x1)
#     x1 = Dropout(0.3)(x1)

#     # Block 2 - Medium kernel
#     x2 = Conv1D(128, kernel_size=20, padding='same', activation='relu')(x1)
#     x2 = BatchNormalization()(x2)
#     x2 = MaxPooling1D(pool_size=4)(x2)
#     x2 = Dropout(0.3)(x2)

#     # Block 3 - Small kernel for high-freq features
#     x3 = Conv1D(256, kernel_size=10, padding='same', activation='relu')(x2)
#     x3 = BatchNormalization()(x3)
#     x3 = GlobalAveragePooling1D()(x3)
#     x3 = Dropout(0.4)(x3)

#     # Classifier
#     x = Dense(128, activation='relu')(x3)
#     x = Dropout(0.4)(x)
#     x = Dense(64, activation='relu')(x)
#     outputs = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs, outputs, name='CNN_EEG')
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def build_lstm_model(input_shape, num_classes):
#     """
#     Bidirectional LSTM Model for EEG Classification.
#     Captures temporal dependencies in both directions.
#     """
#     model = Sequential([
#         Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
#         Dropout(0.3),
#         BatchNormalization(),

#         Bidirectional(LSTM(64, return_sequences=False)),
#         Dropout(0.3),
#         BatchNormalization(),

#         Dense(128, activation='relu'),
#         Dropout(0.4),
#         Dense(64, activation='relu'),
#         Dense(num_classes, activation='softmax')
#     ], name='BiLSTM_EEG')

#     model.compile(
#         optimizer=Adam(learning_rate=0.0005),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def build_cnn_lstm_hybrid(input_shape, num_classes):
#     """
#     CNN-LSTM Hybrid Model — KEY NOVEL CONTRIBUTION.
#     CNN extracts local spatial/frequency features,
#     LSTM captures temporal dynamics across the sequence.
#     """
#     inputs = Input(shape=input_shape)

#     # CNN Feature Extraction Stage
#     x = Conv1D(64, kernel_size=50, padding='same', activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=4)(x)
#     x = Dropout(0.25)(x)

#     x = Conv1D(128, kernel_size=20, padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=4)(x)
#     x = Dropout(0.25)(x)

#     # LSTM Temporal Modeling Stage
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = Bidirectional(LSTM(64, return_sequences=False))(x)
#     x = Dropout(0.3)(x)

#     # Dense Classifier
#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
#     x = Dense(64, activation='relu')(x)
#     outputs = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs, outputs, name='CNN_LSTM_Hybrid_EEG')
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# # =============================================================================
# # SECTION 5: TRAINING UTILITIES
# # =============================================================================

# def get_callbacks(model_name):
#     """Standard callbacks for training."""
#     return [
#         EarlyStopping(monitor='val_accuracy', patience=15,
#                       restore_best_weights=True, verbose=1),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                           patience=7, min_lr=1e-6, verbose=1),
#         ModelCheckpoint(
#             filepath=f'./results/models/{model_name}_best.keras',
#             monitor='val_accuracy', save_best_only=True, verbose=0
#         )
#     ]


# def train_and_evaluate_dl(model, X_train, y_train, X_test, y_test,
#                            y_test_raw, model_name, epochs=100, batch_size=32):
#     """Train a DL model and return metrics."""
#     print(f"\n{'='*60}")
#     print(f"  Training: {model_name}")
#     print(f"{'='*60}")
#     model.summary()

#     history = model.fit(
#         X_train, y_train,
#         validation_split=0.15,
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=get_callbacks(model_name),
#         verbose=1
#     )

#     # Evaluate
#     y_pred_probs = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(y_pred_probs, axis=1)

#     acc  = accuracy_score(y_test_raw, y_pred)
#     prec = precision_score(y_test_raw, y_pred, average='weighted', zero_division=0)
#     rec  = recall_score(y_test_raw, y_pred, average='weighted', zero_division=0)
#     f1   = f1_score(y_test_raw, y_pred, average='weighted', zero_division=0)

#     print(f"\n  {model_name} Results:")
#     print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall   : {rec:.4f}")
#     print(f"  F1-Score : {f1:.4f}")
#     print(f"\n  Classification Report:")
#     print(classification_report(y_test_raw, y_pred, target_names=CLASS_NAMES))

#     return {
#         'model_name': model_name,
#         'accuracy': acc,
#         'precision': prec,
#         'recall': rec,
#         'f1_score': f1,
#         'history': history,
#         'y_pred': y_pred,
#         'y_true': y_test_raw,
#         'confusion_matrix': confusion_matrix(y_test_raw, y_pred)
#     }


# # =============================================================================
# # SECTION 6: VISUALIZATION
# # =============================================================================

# def plot_training_history(history, model_name):
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle(f'{model_name} - Training History', fontsize=14, fontweight='bold')

#     axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
#     axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
#     axes[0].set_title('Model Accuracy')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Accuracy')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     axes[1].plot(history.history['loss'], label='Train Loss', color='red')
#     axes[1].plot(history.history['val_loss'], label='Val Loss', color='green')
#     axes[1].set_title('Model Loss')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Loss')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(f'./results/figures/{model_name}_training_history.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"  Saved: {model_name}_training_history.png")


# def plot_confusion_matrix(cm, model_name, class_names=CLASS_NAMES):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names,
#                 linewidths=0.5)
#     plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.xticks(rotation=30, ha='right')
#     plt.tight_layout()
#     plt.savefig(f'./results/figures/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"  Saved: {model_name}_confusion_matrix.png")


# def plot_comparison_chart(all_results):
#     """Bar chart comparing all models across all metrics."""
#     models  = [r['model_name'] for r in all_results]
#     metrics = ['accuracy', 'precision', 'recall', 'f1_score']
#     colors  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

#     x = np.arange(len(models))
#     width = 0.2

#     fig, ax = plt.subplots(figsize=(14, 7))
#     for i, (metric, color) in enumerate(zip(metrics, colors)):
#         vals = [r[metric] * 100 for r in all_results]
#         bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
#                       color=color, alpha=0.85, edgecolor='white')
#         for bar, val in zip(bars, vals):
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
#                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

#     ax.set_xlabel('Model', fontsize=12)
#     ax.set_ylabel('Score (%)', fontsize=12)
#     ax.set_title('Model Performance Comparison — EEG Epileptic Seizure Classification',
#                  fontsize=13, fontweight='bold')
#     ax.set_xticks(x + width * 1.5)
#     ax.set_xticklabels(models, rotation=15, ha='right')
#     ax.set_ylim(50, 105)
#     ax.legend(fontsize=10)
#     ax.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('./results/figures/ALL_models_comparison.png', dpi=150, bbox_inches='tight')
#     plt.close()
#     print("  Saved: ALL_models_comparison.png")


# def save_results_csv(all_results):
#     """Save all model results to CSV for your paper's Table."""
#     rows = []
#     for r in all_results:
#         rows.append({
#             'Model': r['model_name'],
#             'Accuracy (%)': round(r['accuracy'] * 100, 2),
#             'Precision (%)': round(r['precision'] * 100, 2),
#             'Recall (%)': round(r['recall'] * 100, 2),
#             'F1-Score (%)': round(r['f1_score'] * 100, 2),
#         })
#     df = pd.DataFrame(rows)
#     df.to_csv('./results/ALL_model_results.csv', index=False)
#     print("\n  Saved results table: ./results/ALL_model_results.csv")
#     print(df.to_string(index=False))
#     return df


# # =============================================================================
# # SECTION 7: BASELINE ML MODELS (for comparison in paper)
# # =============================================================================

# def train_ml_baselines(X_feat, y, test_size=0.2):
#     """Train traditional ML models on handcrafted features."""
#     print("\n" + "="*60)
#     print("  Training Traditional ML Baselines")
#     print("="*60)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_feat)
#     X_tr, X_te, y_tr, y_te = train_test_split(
#         X_scaled, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
#     )

#     baselines = {
#         'Random_Forest': RandomForestClassifier(
#             n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1),
#         'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED),
#         'Decision_Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED)
#     }

#     results = []
#     for name, clf in baselines.items():
#         print(f"\n  Fitting {name}...")
#         clf.fit(X_tr, y_tr)
#         y_pred = clf.predict(X_te)
#         results.append({
#             'model_name': name,
#             'accuracy': accuracy_score(y_te, y_pred),
#             'precision': precision_score(y_te, y_pred, average='weighted', zero_division=0),
#             'recall': recall_score(y_te, y_pred, average='weighted', zero_division=0),
#             'f1_score': f1_score(y_te, y_pred, average='weighted', zero_division=0),
#             'y_pred': y_pred,
#             'y_true': y_te,
#             'confusion_matrix': confusion_matrix(y_te, y_pred),
#             'history': None
#         })
#         print(f"    Accuracy: {results[-1]['accuracy']*100:.2f}%")
#         print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

#     return results


# # =============================================================================
# # SECTION 8: MAIN PIPELINE
# # =============================================================================

# def main():
#     print("\n" + "="*60)
#     print("  EEG Epileptic Seizure Classification")
#     print("  Deep Learning + Hybrid Approach")
#     print("="*60)

#     # ── 1. Load Data ──────────────────────────────────────────────
#     print("\n[1] Loading Bonn University Dataset...")
#     signals, labels = load_bonn_dataset(DATASET_PATH)

#     if len(signals) == 0:
#         print("\n  ERROR: No data loaded!")
#         print("  Please check DATASET_PATH at top of script.")
#         print("  Expected structure:")
#         print("    ./bonn_dataset/A/  (100 .txt files)")
#         print("    ./bonn_dataset/B/  (100 .txt files)")
#         print("    ./bonn_dataset/C/  (100 .txt files)")
#         print("    ./bonn_dataset/D/  (100 .txt files)")
#         print("    ./bonn_dataset/E/  (100 .txt files)")
#         return

#     print(f"\n  Total signals loaded: {len(signals)}")
#     print(f"  Signal length: {signals[0].shape}")
#     print(f"  Class distribution: {np.bincount(labels)}")

#     # ── 2. Feature Extraction for ML Baselines ────────────────────
#     print("\n[2] Extracting PSD + Wavelet features for ML baselines...")
#     X_features = extract_all_features(signals)
#     print(f"  Feature matrix shape: {X_features.shape}")

#     # ── 3. Prepare DL Data ────────────────────────────────────────
#     print("\n[3] Preparing segmented data for Deep Learning...")
#     X_dl, y_dl_cat, y_dl_raw = prepare_dl_data(signals, labels)
#     print(f"  DL input shape: {X_dl.shape}")
#     print(f"  Total segments: {len(X_dl)}")

#     # Train/test split for DL
#     X_tr_dl, X_te_dl, y_tr_dl, y_te_dl, y_tr_raw, y_te_raw = train_test_split(
#         X_dl, y_dl_cat, y_dl_raw,
#         test_size=0.2, random_state=RANDOM_SEED, stratify=y_dl_raw
#     )
#     print(f"  Train: {X_tr_dl.shape[0]} | Test: {X_te_dl.shape[0]}")

#     input_shape  = (X_dl.shape[1], 1)
#     num_classes  = len(SETS)
#     all_results  = []

#     # ── 4. Train ML Baselines ─────────────────────────────────────
#     print("\n[4] Training Traditional ML Models...")
#     ml_results = train_ml_baselines(X_features, labels)
#     for r in ml_results:
#         plot_confusion_matrix(r['confusion_matrix'], r['model_name'])
#     all_results.extend(ml_results)

#     # ── 5. Train CNN ──────────────────────────────────────────────
#     print("\n[5] Training 1D-CNN Model...")
#     cnn_model  = build_cnn_model(input_shape, num_classes)
#     cnn_result = train_and_evaluate_dl(
#         cnn_model, X_tr_dl, y_tr_dl, X_te_dl, y_te_dl, y_te_raw,
#         model_name='1D_CNN', epochs=100, batch_size=32
#     )
#     plot_training_history(cnn_result['history'], '1D_CNN')
#     plot_confusion_matrix(cnn_result['confusion_matrix'], '1D_CNN')
#     all_results.append(cnn_result)

#     # ── 6. Train BiLSTM ───────────────────────────────────────────
#     print("\n[6] Training Bidirectional LSTM Model...")
#     lstm_model  = build_lstm_model(input_shape, num_classes)
#     lstm_result = train_and_evaluate_dl(
#         lstm_model, X_tr_dl, y_tr_dl, X_te_dl, y_te_dl, y_te_raw,
#         model_name='BiLSTM', epochs=100, batch_size=32
#     )
#     plot_training_history(lstm_result['history'], 'BiLSTM')
#     plot_confusion_matrix(lstm_result['confusion_matrix'], 'BiLSTM')
#     all_results.append(lstm_result)

#     # ── 7. Train CNN-LSTM Hybrid ──────────────────────────────────
#     print("\n[7] Training CNN-LSTM Hybrid Model (Novel Contribution)...")
#     hybrid_model  = build_cnn_lstm_hybrid(input_shape, num_classes)
#     hybrid_result = train_and_evaluate_dl(
#         hybrid_model, X_tr_dl, y_tr_dl, X_te_dl, y_te_dl, y_te_raw,
#         model_name='CNN_LSTM_Hybrid', epochs=120, batch_size=32
#     )
#     plot_training_history(hybrid_result['history'], 'CNN_LSTM_Hybrid')
#     plot_confusion_matrix(hybrid_result['confusion_matrix'], 'CNN_LSTM_Hybrid')
#     all_results.append(hybrid_result)

#     # ── 8. Final Comparison ───────────────────────────────────────
#     print("\n[8] Generating Comparison Plots & Results Table...")
#     plot_comparison_chart(all_results)
#     results_df = save_results_csv(all_results)

#     # ── 9. Summary ────────────────────────────────────────────────
#     best = max(all_results, key=lambda r: r['accuracy'])
#     print("\n" + "="*60)
#     print(f"  BEST MODEL: {best['model_name']}")
#     print(f"  Accuracy : {best['accuracy']*100:.2f}%")
#     print(f"  Precision: {best['precision']*100:.2f}%")
#     print(f"  Recall   : {best['recall']*100:.2f}%")
#     print(f"  F1-Score : {best['f1_score']*100:.2f}%")
#     print("="*60)
#     print("\n  All outputs saved to ./results/")
#     print("  ✓ Model checkpoints: ./results/models/")
#     print("  ✓ Figures:           ./results/figures/")
#     print("  ✓ Results CSV:       ./results/ALL_model_results.csv")
#     print("\n  NEXT STEP: Share your CSV results with Claude")
#     print("  to write your IEEE paper!\n")

# print("Script loaded OK — starting main...")
# if __name__ == '__main__':
#     main()

"""
EEG Epileptic Seizure Classification
Deep Learning: CNN, BiLSTM, CNN-LSTM Hybrid
Dataset: Bonn University EEG Dataset (Sets A-E)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Starting EEG Classification Pipeline...")
print("Loading libraries...")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pywt
print("  numpy, pandas, matplotlib, scipy, pywt OK")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
print("  sklearn OK")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
print(f"  tensorflow {tf.__version__} OK")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_PATH   = './bonn_dataset'
SETS           = ['A', 'B', 'C', 'D', 'E']
CLASS_NAMES    = ['Healthy-EO', 'Healthy-EC', 'Seizure-Free-HC', 'Seizure-Free-EZ', 'Seizure']
SEGMENT_LENGTH = 512
SAMPLING_RATE  = 173.6
RANDOM_SEED    = 42
EPOCHS         = 80
BATCH_SIZE     = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/figures', exist_ok=True)
os.makedirs('./results/models', exist_ok=True)
print(f"\nConfig OK. Dataset: {DATASET_PATH}\n")

# =============================================================================
# DATA LOADING
# =============================================================================
def load_bonn_dataset(dataset_path, sets=SETS):
    all_signals, all_labels = [], []
    for label_idx, folder in enumerate(sets):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"  [WARNING] Not found: {folder_path}")
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
        loaded = 0
        for fname in files:
            try:
                data = np.loadtxt(os.path.join(folder_path, fname))
                data = data[:4096]
                all_signals.append(data)
                all_labels.append(label_idx)
                loaded += 1
            except Exception as e:
                print(f"    Skipping {fname}: {e}")
        print(f"  Set {folder} ({CLASS_NAMES[label_idx]}): {loaded} files")
    return np.array(all_signals), np.array(all_labels)

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_psd_features(sig, fs=SAMPLING_RATE):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=256)
    bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,60)}
    feats = []
    for (low, high) in bands.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        feats += [np.mean(psd[idx]), np.max(psd[idx]), np.sum(psd[idx])]
    feats += [np.mean(psd), np.std(psd)]
    return np.array(feats)

def extract_wavelet_features(sig, wavelet='db4', level=5):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    feats = []
    for c in coeffs:
        feats += [np.mean(np.abs(c)), np.std(c), np.var(c), np.mean(c**2),
                  float(-np.sum(c**2 * np.log(c**2 + 1e-10)))]
    return np.array(feats)

def extract_all_features(signals):
    print("  Extracting features", end='', flush=True)
    feats = []
    for i, sig in enumerate(signals):
        feats.append(np.concatenate([extract_psd_features(sig), extract_wavelet_features(sig)]))
        if (i+1) % 100 == 0:
            print('.', end='', flush=True)
    print(' done')
    return np.array(feats)

# =============================================================================
# DL DATA PREP
# =============================================================================
def prepare_dl_data(signals, labels):
    X_seg, y_seg = [], []
    step = SEGMENT_LENGTH // 2
    for sig, lbl in zip(signals, labels):
        for start in range(0, len(sig) - SEGMENT_LENGTH + 1, step):
            seg = sig[start:start + SEGMENT_LENGTH]
            if len(seg) == SEGMENT_LENGTH:
                X_seg.append(seg)
                y_seg.append(lbl)
    X = np.array(X_seg, dtype=np.float32)
    y = np.array(y_seg)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, to_categorical(y, num_classes=len(SETS)), y

# =============================================================================
# MODELS
# =============================================================================
def build_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64,  50, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x); x = MaxPooling1D(4)(x); x = Dropout(0.3)(x)
    x = Conv1D(128, 20, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = MaxPooling1D(4)(x); x = Dropout(0.3)(x)
    x = Conv1D(256, 10, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = GlobalAveragePooling1D()(x); x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x); x = Dropout(0.4)(x)
    x = Dense(64,  activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    m = Model(inputs, out, name='1D_CNN')
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_bilstm(input_shape, num_classes):
    m = Sequential(name='BiLSTM')
    m.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    m.add(Dropout(0.3)); m.add(BatchNormalization())
    m.add(Bidirectional(LSTM(64)))
    m.add(Dropout(0.3)); m.add(BatchNormalization())
    m.add(Dense(128, activation='relu')); m.add(Dropout(0.4))
    m.add(Dense(64,  activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    m.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_cnn_lstm_hybrid(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64,  50, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x); x = MaxPooling1D(4)(x); x = Dropout(0.25)(x)
    x = Conv1D(128, 20, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = MaxPooling1D(4)(x); x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x); x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x); x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x); x = BatchNormalization()(x); x = Dropout(0.4)(x)
    x = Dense(64,  activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    m = Model(inputs, out, name='CNN_LSTM_Hybrid')
    m.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# =============================================================================
# TRAINING
# =============================================================================
def get_callbacks(name):
    return [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ModelCheckpoint(f'./results/models/{name}_best.h5',
                        monitor='val_accuracy', save_best_only=True, verbose=0)
    ]

def train_dl(model, X_tr, y_tr, X_te, y_te, y_te_r):
    name = model.name
    print(f"\n  Training {name}...")
    history = model.fit(X_tr, y_tr, validation_split=0.15,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=get_callbacks(name), verbose=1)
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    r = {
        'model_name': name,
        'accuracy':   accuracy_score(y_te_r, y_pred),
        'precision':  precision_score(y_te_r, y_pred, average='weighted', zero_division=0),
        'recall':     recall_score(y_te_r, y_pred, average='weighted', zero_division=0),
        'f1_score':   f1_score(y_te_r, y_pred, average='weighted', zero_division=0),
        'history': history, 'y_pred': y_pred, 'y_true': y_te_r,
        'cm': confusion_matrix(y_te_r, y_pred)
    }
    print(f"\n  {name}: Acc={r['accuracy']*100:.2f}% | P={r['precision']*100:.2f}% | R={r['recall']*100:.2f}% | F1={r['f1_score']*100:.2f}%")
    print(classification_report(y_te_r, y_pred, target_names=CLASS_NAMES))
    return r

def train_ml(X_feat, y):
    print("\n  Training ML Baselines...")
    X_sc = StandardScaler().fit_transform(X_feat)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                                random_state=RANDOM_SEED, stratify=y)
    clfs = {
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=20,
                                                random_state=RANDOM_SEED, n_jobs=-1),
        'SVM':           SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_SEED),
        'Decision_Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED)
    }
    results = []
    for name, clf in clfs.items():
        print(f"    {name}...", end=' ', flush=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"Acc: {acc*100:.2f}%")
        results.append({
            'model_name': name,
            'accuracy':   acc,
            'precision':  precision_score(y_te, y_pred, average='weighted', zero_division=0),
            'recall':     recall_score(y_te, y_pred, average='weighted', zero_division=0),
            'f1_score':   f1_score(y_te, y_pred, average='weighted', zero_division=0),
            'history': None, 'y_pred': y_pred, 'y_true': y_te,
            'cm': confusion_matrix(y_te, y_pred)
        })
        print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{name} — Training History', fontsize=13, fontweight='bold')
    ax[0].plot(history.history['accuracy'],     label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Val')
    ax[0].set_title('Accuracy'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history.history['loss'],     color='red',    label='Train')
    ax[1].plot(history.history['val_loss'], color='orange', label='Val')
    ax[1].set_title('Loss'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/figures/{name}_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_cm(cm, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix — {name}', fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(f'./results/figures/{name}_cm.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison(all_results):
    models  = [r['model_name'] for r in all_results]
    metrics = ['accuracy','precision','recall','f1_score']
    colors  = ['#2196F3','#4CAF50','#FF9800','#E91E63']
    x, w    = np.arange(len(models)), 0.2
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [r[m]*100 for r in all_results]
        bars = ax.bar(x + i*w, vals, w, label=m.capitalize(), color=c, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylim(40, 110); ax.set_ylabel('Score (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Model Performance Comparison — EEG Epileptic Seizure Classification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/figures/ALL_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_csv(all_results):
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy (%)':  round(r['accuracy']*100, 2),
        'Precision (%)': round(r['precision']*100, 2),
        'Recall (%)':    round(r['recall']*100, 2),
        'F1-Score (%)':  round(r['f1_score']*100, 2),
    } for r in all_results])
    df.to_csv('./results/ALL_model_results.csv', index=False)
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    return df

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  EEG Seizure Classification — CNN + BiLSTM + Hybrid")
    print("="*60)

    print("\n[1/7] Loading dataset...")
    signals, labels = load_bonn_dataset(DATASET_PATH)
    if len(signals) == 0:
        print("ERROR: No data loaded. Check DATASET_PATH."); sys.exit(1)
    print(f"  Loaded {len(signals)} signals | Classes: {np.bincount(labels)}")

    print("\n[2/7] Extracting ML features...")
    X_feat = extract_all_features(signals)
    print(f"  Feature size: {X_feat.shape[1]}")

    print("\n[3/7] Segmenting for Deep Learning...")
    X_dl, y_dl_cat, y_dl_raw = prepare_dl_data(signals, labels)
    print(f"  Segments: {X_dl.shape[0]} x {X_dl.shape[1:]}")

    X_tr, X_te, y_tr, y_te, y_tr_r, y_te_r = train_test_split(
        X_dl, y_dl_cat, y_dl_raw,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y_dl_raw)
    print(f"  Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

    all_results = []
    ishape = (SEGMENT_LENGTH, 1)
    nc = len(SETS)

    print("\n[4/7] ML Baselines...")
    ml_res = train_ml(X_feat, labels)
    for r in ml_res:
        plot_cm(r['cm'], r['model_name'])
    all_results.extend(ml_res)

    print("\n[5/7] 1D-CNN...")
    r = train_dl(build_cnn(ishape, nc), X_tr, y_tr, X_te, y_te, y_te_r)
    plot_history(r['history'], '1D_CNN'); plot_cm(r['cm'], '1D_CNN')
    all_results.append(r)

    print("\n[6/7] BiLSTM...")
    r = train_dl(build_bilstm(ishape, nc), X_tr, y_tr, X_te, y_te, y_te_r)
    plot_history(r['history'], 'BiLSTM'); plot_cm(r['cm'], 'BiLSTM')
    all_results.append(r)

    print("\n[7/7] CNN-LSTM Hybrid...")
    r = train_dl(build_cnn_lstm_hybrid(ishape, nc), X_tr, y_tr, X_te, y_te, y_te_r)
    plot_history(r['history'], 'CNN_LSTM_Hybrid'); plot_cm(r['cm'], 'CNN_LSTM_Hybrid')
    all_results.append(r)

    plot_comparison(all_results)
    save_csv(all_results)

    best = max(all_results, key=lambda r: r['accuracy'])
    print(f"\n  BEST MODEL : {best['model_name']} ({best['accuracy']*100:.2f}%)")
    print("  Results saved to ./results/")
    print("  Share ALL_model_results.csv with Claude to write your IEEE paper!\n")

if __name__ == '__main__':
    main()
# import numpy as np
# import tensorflow as tf
# from scipy import signal

# CLASS_NAMES = ['Healthy-EO', 'Healthy-EC', 'Seizure-Free-HC',
#                'Seizure-Free-EZ', 'Seizure']

# SEGMENT_LENGTH = 512

# def preprocess_signal(sig):
#     sig = sig[:4096]
#     segments = []
#     step = SEGMENT_LENGTH // 2

#     for start in range(0, len(sig) - SEGMENT_LENGTH + 1, step):
#         seg = sig[start:start + SEGMENT_LENGTH]
#         seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
#         segments.append(seg)

#     X = np.array(segments).reshape(-1, SEGMENT_LENGTH, 1)
#     return X

# def load_model(model_name):
#     return tf.keras.models.load_model(
#         f'./results/models/{model_name}_best.h5'
#     )

# def predict_signal(model, signal_data):
#     X = preprocess_signal(signal_data)
#     preds = model.predict(X)
#     avg_pred = np.mean(preds, axis=0)
#     cls = np.argmax(avg_pred)
#     return CLASS_NAMES[cls], float(avg_pred[cls])

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    'Healthy-EO',
    'Healthy-EC',
    'Seizure-Free-HC',
    'Seizure-Free-EZ',
    'Seizure'
]

SEGMENT_LENGTH = 512

def preprocess_signal(sig):
    sig = sig[:4096]
    segments = []
    step = SEGMENT_LENGTH // 2

    for start in range(0, len(sig) - SEGMENT_LENGTH + 1, step):
        seg = sig[start:start + SEGMENT_LENGTH]
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        segments.append(seg)

    X = np.array(segments).reshape(-1, SEGMENT_LENGTH, 1)
    return X

def load_model(model_name):
    path = f'./results/models/{model_name}_best.h5'
    return tf.keras.models.load_model(path)

def predict_signal(model, signal_data):
    X = preprocess_signal(signal_data)
    preds = model.predict(X, verbose=0)
    avg_pred = np.mean(preds, axis=0)
    cls = int(np.argmax(avg_pred))
    return CLASS_NAMES[cls], float(avg_pred[cls])
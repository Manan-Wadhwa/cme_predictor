# Refactored LSTM + ROCKET Classification Pipeline for HALO Prediction with Enhanced Logging and Evaluation

#%%
# ===============================
# 📦 Imports & Utilities
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.datatypes._panel._convert import from_2d_array_to_nested

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def log_metrics(log_file, metrics):
    with open(log_file, 'w') as f:
        json.dump(metrics, f, indent=4)

#%%
# ===============================
# 📊 Load & Preprocess Data
# ===============================
print("Loading data...")
df = pd.read_csv('merged_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Processing full dataset with {len(df)} rows...")

print("Preprocessing timestamps and missing values...")
df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'])
df.replace(-1e31, np.nan, inplace=True)
df = df.sort_values('epoch_for_cdf_mod')
df.dropna(thresh=len(df.columns) * 0.7, inplace=True)
df = df.ffill()
df = df.bfill()

print("Feature engineering...")
df['hour'] = df['epoch_for_cdf_mod'].dt.hour
df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday
columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']
df.drop(columns=[c for c in columns_to_remove if c in df.columns], inplace=True)

print(f"After preprocessing: {df.shape}")
print(f"HALO distribution:\n{df['HALO'].value_counts()}")

#%%
# ===============================
# 📈 Feature & Target Prep
# ===============================
print("Preparing features and targets...")
feature_columns = [col for col in df.columns if col != 'HALO']
target_column = 'HALO'

X_raw = df[feature_columns].values
y = df[target_column].values

print("Splitting train/test...")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling features...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

seq_length = 50

def create_sequences_efficient(data, target, seq_length):
    n_samples = len(data) - seq_length
    X_seq = np.zeros((n_samples, seq_length, data.shape[1]))
    y_seq = np.zeros(n_samples)
    for i in range(n_samples):
        X_seq[i] = data[i:i+seq_length]
        y_seq[i] = target[i+seq_length]
    return X_seq, y_seq

print("Creating sequences...")
X_train_seq, y_train_seq = create_sequences_efficient(X_train_scaled, y_train_raw, seq_length)
X_test_seq, y_test_seq = create_sequences_efficient(X_test_scaled, y_test_raw, seq_length)

print("One-hot encoding targets...")
y_train_cat = to_categorical(y_train_seq)
y_test_cat = to_categorical(y_test_seq)
num_classes = y_train_cat.shape[1]

#%%
# ===============================
# 🧪 Class Weights
# ===============================
print("Computing class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

#%%
# ===============================
# 🧠 LSTM Model Builder
# ===============================
print("Building LSTM model...")
def build_lstm_model(optimizer='adam', dropout=0.3):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, X_train_seq.shape[2])),
        Dropout(dropout),
        LSTM(32),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
model = build_lstm_model()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_lstm_model.h5", save_best_only=True)
]

print("Training base LSTM model...")
history = model.fit(
    X_train_seq, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("Saving LSTM training curve...")
pd.DataFrame(history.history).to_csv("logs/lstm_training_curve.csv", index=False)

#%%
# ===============================
# 📉 Evaluation - LSTM
# ===============================
print("Evaluating base LSTM...")
base_pred = np.argmax(model.predict(X_test_seq), axis=1)
y_true = y_test_seq.astype(int)

base_metrics = {
    'accuracy': accuracy_score(y_true, base_pred),
    'precision': precision_score(y_true, base_pred, average='weighted'),
    'recall': recall_score(y_true, base_pred, average='weighted'),
    'f1': f1_score(y_true, base_pred, average='weighted')
}
log_metrics(os.path.join(log_dir, "base_lstm_metrics.json"), base_metrics)
print("\nClassification Report:\n", classification_report(y_true, base_pred))

print("Saving LSTM confusion matrix plot...")
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, base_pred), annot=True, fmt='d', cmap='Blues')
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("logs/lstm_confusion_matrix.png")
plt.close()

#%%
# ===============================
# 🚀 ROCKET + RidgeClassifier
# ===============================
print("Running ROCKET + RidgeClassifier...")
X_nested = from_2d_array_to_nested(X_train_seq[:, :, 0])  # Single channel
rocket = Rocket(num_kernels=10000, random_state=42)
X_transformed = rocket.fit_transform(X_nested)
X_test_nested = from_2d_array_to_nested(X_test_seq[:, :, 0])
X_test_transformed = rocket.transform(X_test_nested)

from sklearn.linear_model import RidgeClassifierCV
ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
ridge.fit(X_transformed, y_train_seq)
rocket_pred = ridge.predict(X_test_transformed)

rocket_metrics = {
    'accuracy': accuracy_score(y_test_seq, rocket_pred),
    'precision': precision_score(y_test_seq, rocket_pred, average='weighted'),
    'recall': recall_score(y_test_seq, rocket_pred, average='weighted'),
    'f1': f1_score(y_test_seq, rocket_pred, average='weighted')
}
log_metrics(os.path.join(log_dir, "rocket_metrics.json"), rocket_metrics)
print("\nROCKET Classification Report:\n", classification_report(y_test_seq, rocket_pred))

#%%
# ===============================
# 🔍 Hyperparameter Tuning - LSTM
# ===============================
print("Running GridSearchCV on LSTM...")
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout': [0.2, 0.3, 0.4],
    'epochs': [10],
    'batch_size': [64]
}

grid = GridSearchCV(KerasClassifier(model=build_lstm_model, verbose=0), param_grid, cv=2)
grid.fit(X_train_seq, y_train_cat)

best_model = grid.best_estimator_
hpo_pred = np.argmax(best_model.predict(X_test_seq), axis=1)

hpo_metrics = {
    'accuracy': accuracy_score(y_true, hpo_pred),
    'precision': precision_score(y_true, hpo_pred, average='weighted'),
    'recall': recall_score(y_true, hpo_pred, average='weighted'),
    'f1': f1_score(y_true, hpo_pred, average='weighted')
}
log_metrics(os.path.join(log_dir, "hpo_lstm_metrics.json"), hpo_metrics)
print("\nHPO LSTM Report:\n", classification_report(y_true, hpo_pred))

#%%
# ===============================
# 📦 Final Summary
# ===============================
print("\nSaved logs to logs/:")
print("- base_lstm_metrics.json")
print("- hpo_lstm_metrics.json")
print("- rocket_metrics.json")
print("- lstm_training_curve.csv")
print("- lstm_confusion_matrix.png")
# %%

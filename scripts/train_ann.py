"""
Reproducible Training and Export Script for TensorFlow/Keras Fraud ANN.

This script trains a Deep Artificial Neural Network on the preprocessed Kaggle
Credit Card Fraud dataset (data/processed/train_processed.csv) and evaluates
it against the holdout test set (data/processed/test_processed.csv).

Exports:
- models/fraud_ann_model.keras
- outputs/metrics/deep_learning_results.csv
- outputs/metrics/ml_dl_comparison.csv
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_export_ann(epochs: int = 15, batch_size: int = 2048):
    train_file = DATA_DIR / "train_processed.csv"
    test_file = DATA_DIR / "test_processed.csv"

    if not train_file.is_file() or not test_file.is_file():
        print(f"[ERROR] Processed data files not found in {DATA_DIR}.")
        print("Expected train_processed.csv and test_processed.csv.")
        sys.exit(1)

    print(f"Loading preprocessed training data from {train_file.name}...")
    train_df = pd.read_csv(train_file)
    print(f"Loading preprocessed testing data from {test_file.name}...")
    test_df = pd.read_csv(test_file)

    feature_cols = [c for c in train_df.columns if c not in ["Class", "class"]]
    target_col = "Class" if "Class" in train_df.columns else "class"

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    n_samples, n_features = X_train.shape
    fraud_count = int(np.sum(y_train == 1))
    legit_count = int(np.sum(y_train == 0))
    print(f"Train Shape: ({n_samples}, {n_features}) | Legitimate: {legit_count}, Fraud: {fraud_count}")

    # Compute class weights for severe class imbalance
    weight_legit = (1 / legit_count) * (n_samples / 2.0)
    weight_fraud = (1 / fraud_count) * (n_samples / 2.0)
    class_weights = {0: weight_legit, 1: weight_fraud}
    print(f"Computed Class Weights: Legit={weight_legit:.4f}, Fraud={weight_fraud:.4f}")

    # Build Keras Architecture
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, mode="min")
    ]

    print("\n--- Training Deep Artificial Neural Network ---")
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = round(time.time() - start_time, 2)
    print(f"Training completed in {training_time} seconds.")

    # Model Evaluation on holdout test set
    print("\n--- Evaluating ANN on Holdout Test Set ---")
    y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_pred_prob))
    pr_auc = float(average_precision_score(y_test, y_pred_prob))
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Export Model
    model_save_path = MODELS_DIR / "fraud_ann_model.keras"
    model.save(str(model_save_path))
    print(f"\n[SUCCESS] Saved Keras model to: {model_save_path}")

    # Export DL Metrics CSV
    dl_metrics = pd.DataFrame([{
        "Model": "Artificial Neural Network (ANN)",
        "Architecture": "Dense(64)-BatchNorm-Dense(32)-BatchNorm-Dense(16)-Sigmoid",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Training Time (sec)": training_time,
    }])
    dl_results_path = OUTPUTS_DIR / "deep_learning_results.csv"
    dl_metrics.to_csv(dl_results_path, index=False)
    print(f"[SUCCESS] Exported deep learning metrics to: {dl_results_path}")

    # Combine with ML metrics for ml_dl_comparison.csv
    ml_metrics_path = OUTPUTS_DIR / "ml_model_comparison.csv"
    if ml_metrics_path.is_file():
        ml_df = pd.read_csv(ml_metrics_path)
        combined_rows = ml_df.to_dict(orient="records")
        combined_rows.append({
            "Model": "Deep ANN (Keras)",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "Training Time (sec)": training_time,
        })
        combined_df = pd.DataFrame(combined_rows)
        combined_path = OUTPUTS_DIR / "ml_dl_comparison.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"[SUCCESS] Exported combined ML vs DL comparison to: {combined_path}")


if __name__ == "__main__":
    train_and_export_ann()

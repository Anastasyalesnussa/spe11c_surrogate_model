import os
import numpy as np
import joblib
from typing import Tuple, Optional


DATA_PROCESSED_DIR = os.path.join("data", "processed")


def load_train_test() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-split training and testing datasets.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = np.load(os.path.join(DATA_PROCESSED_DIR, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(DATA_PROCESSED_DIR, "X_test_scaled.npy"))
    y_train = np.load(os.path.join(DATA_PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def load_full_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load full processed dataset (X and y before split), if needed for retraining.
    Returns:
        X, y
    """
    X = np.load(os.path.join(DATA_PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_PROCESSED_DIR, "y.npy"))
    return X, y


def load_4D_ml_ready() -> Optional[np.ndarray]:
    """
    Load the 4D spatial dataset for visualization/inference.
    Returns:
        numpy array of shape (samples, x_dim, y_dim, z_dim)
    """
    file_path = os.path.join(DATA_PROCESSED_DIR, "ml_ready_4D_dataset.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print("ml_ready_4D_dataset.npy not found.")
        return None


def load_y_scaler():
    """
    Load the scaler used for target variable.
    Returns:
        scaler object
    """
    scaler_path = os.path.join("models", "y_scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        print("y_scaler.pkl not found.")
        return None

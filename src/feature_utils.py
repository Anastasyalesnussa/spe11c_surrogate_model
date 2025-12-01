import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def scale_features(X_train, X_test, save=True):
    """
    Fit StandardScaler on training features and apply it to both
    training and test sets. Saves the scaler to /models if save=True.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save:
        joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
        print("Saved → models/feature_scaler.pkl")

    return X_train_scaled, X_test_scaled
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def scale_features(X_train, X_test, save=True):
    """
    Fit StandardScaler on training features and apply it to both
    training and test sets. Saves the scaler to /models if save=True.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save:
        joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
        print("Saved → models/feature_scaler.pkl")

    return X_train_scaled, X_test_scaled

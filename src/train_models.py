import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_scaled_data():
    X_train = np.load("data/processed/X_train_scaled.npy")
    X_test = np.load("data/processed/X_test_scaled.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"  -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rmse, r2


def save_model(model, filename):
    file_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, file_path)
    print(f"Saved → {file_path}")


def train_all_models():
    X_train, X_test, y_train, y_test = load_scaled_data()

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "MLP_Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            alpha=0.0005,              
            batch_size=64,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=42
        )
    }

    results = {}

    print("\n===== Training Baseline Models =====\n")
    for name, model in models.items():
        print(f"> Training {name} ...")
        model.fit(X_train, y_train)
        rmse, r2 = evaluate(name, model, X_test, y_test)
        results[name] = {"RMSE": rmse, "R2": r2}

        if name == "MLP_Regressor":
            save_model(model, "MLP_baseline.pkl")     
        else:
            save_model(model, f"{name}_baseline.pkl")

    print("\n===== Baseline Model Training Completed =====")
    return results


if __name__ == "__main__":
    results = train_all_models()

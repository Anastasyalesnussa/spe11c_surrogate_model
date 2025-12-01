import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
DATA_DIR = "data/processed"

def load_scaled_data():
    X_test = np.load(os.path.join(DATA_DIR, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test

def evaluate_all_models():
    X_test, y_test = load_scaled_data()
    results = []

    y_scaler_path = os.path.join(MODELS_DIR, "y_scaler.pkl")
    y_scaler = joblib.load(y_scaler_path) if os.path.exists(y_scaler_path) else None

    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith((".pkl", ".h5", ".keras"))
        and "scaler" not in f.lower()
        and "readme" not in f.lower()
    ]

    for model_file in model_files:
        print(f"> Evaluating {model_file} ...")
        model_path = os.path.join(MODELS_DIR, model_file)

        if model_file.endswith(".pkl"):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

        elif model_file.endswith((".h5", ".keras")):
            model = load_model(model_path, compile=False)
            y_pred_scaled = model.predict(X_test).flatten()
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
            else:
                y_pred = y_pred_scaled

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        model_name = (
            model_file.replace(".pkl", " (sklearn)")
                      .replace(".h5", " (keras)")
                      .replace(".keras", " (keras)")
        )

        NAME_MAPPING = {
            "LinearRegression_baseline (sklearn)": "LinearRegression_baseline",
            "RandomForest_baseline (sklearn)": "RandomForest_baseline",
            "GradientBoosting_baseline (sklearn)": "GradientBoosting_baseline",
            "MLP_baseline (sklearn)": "MLPRegressor_baseline",
            "MLP_baseline (keras)": "NN_baseline",
            "Surrogate_MLP (keras)": "Surrogate_NN_Final"
        }


        model_name = NAME_MAPPING.get(model_name, model_name)

        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "R²": r2
        })

    df = pd.DataFrame(results).sort_values(by="RMSE")
    df.to_csv("reports/model_performance.csv", index=False)

    print("\n✔ Evaluation complete → reports/model_performance.csv")
    return df


if __name__ == "__main__":
    print("\n===== Evaluating Saved Models =====\n")
    df_results = evaluate_all_models()
    print(df_results)

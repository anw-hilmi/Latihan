import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    print("--- Memulai Proses ---")
    
    # Memuat Data
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    print(f"Memuat data dari: {file_path}")
    data = pd.read_csv(file_path)

    # Split Data
    print("Membagi data menjadi set train dan test...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]
    
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    print(f"Parameter: n_estimators={n_estimators}, max_depth={max_depth}")

    with mlflow.start_run():
        print("Mulai logging ke MLflow...")
        
        # Inisialisasi dan Training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        print("Sedang melatih model (Fitting)...")
        model.fit(X_train, y_train)

        # Prediksi
        print("Melakukan prediksi pada data test...")
        predicted_qualities = model.predict(X_test)

        # Log Model
        print("Menyimpan model ke MLflow Artifacts...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Log Metrics
        print("Menghitung dan mencatat metrik akurasi...")
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Selesai! Akurasi Model: {accuracy:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    print("--- Proses Selesai ---")
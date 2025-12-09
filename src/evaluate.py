import argparse
import json
import os
import tarfile
from glob import glob

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--test-data", type=str, default="/opt/ml/input/data/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output")

    return parser.parse_args()


def load_data(path: str):
    if os.path.isdir(path):
        frames = []
        for fname in os.listdir(path):
            if fname.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(path, fname)))
        if not frames:
            raise ValueError(f"No CSV files found in directory: {path}")
        data = pd.concat(frames, ignore_index=True)
    else:
        data = pd.read_csv(path)

    return data


def find_or_extract_model(model_dir: str) -> str:
    """
    Look for a *.joblib model in model_dir.
    If not found, look for a *.tar.gz, extract it, then search again.
    Returns the path to the model.joblib file.
    """
    # 1) Check for existing .joblib
    joblib_candidates = glob(os.path.join(model_dir, "**", "*.joblib"), recursive=True)
    if joblib_candidates:
        print(f"Found existing model joblib: {joblib_candidates[0]}")
        return joblib_candidates[0]

    # 2) Try to find a model.tar.gz
    tar_candidates = glob(os.path.join(model_dir, "*.tar.gz"))
    if not tar_candidates:
        raise FileNotFoundError(f"No .joblib or .tar.gz model files found in {model_dir}")

    tar_path = tar_candidates[0]
    print(f"No joblib found, extracting from tarball: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)

    # 3) Search again for .joblib after extraction
    joblib_candidates = glob(os.path.join(model_dir, "**", "*.joblib"), recursive=True)
    if not joblib_candidates:
        raise FileNotFoundError(
            f"After extracting {tar_path}, no .joblib model found in {model_dir}"
        )

    print(f"Found model joblib after extraction: {joblib_candidates[0]}")
    return joblib_candidates[0]


def main():
    args = parse_args()

    model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)
    test_path = os.environ.get("SM_CHANNEL_TEST", args.test_data)
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model ----
    model_path = find_or_extract_model(model_dir)
    model = joblib.load(model_path)

    # ---- Load test data ----
    df = load_data(test_path)
    X = df[["size_sqft", "num_rooms", "age_years"]]
    y = df["price"]

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    print(f"Test MSE: {mse:.4f}")

    # ---- Save metrics to JSON ----
    metrics = {
        "mse": mse,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

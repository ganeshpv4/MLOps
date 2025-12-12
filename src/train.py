import argparse
import os
import sys
import traceback

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser()

    # For local runs
    parser.add_argument("--train-data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")

    return parser.parse_args()


def load_training_data(train_path: str):
    """
    If train_path is a directory (SageMaker style), read all CSVs inside.
    If it's a file, read that file.
    """
    if os.path.isdir(train_path):
        # read all .csv files in the directory
        frames = []
        for fname in os.listdir(train_path):
            if fname.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(train_path, fname)))
        if not frames:
            raise ValueError(f"No CSV files found in directory: {train_path}")
        data = pd.concat(frames, ignore_index=True)
    else:
        data = pd.read_csv(train_path)

    return data


def main():
    try:
        args = parse_args()

        # Resolve paths (local vs SageMaker)
        train_path = args.train_data
        model_dir = args.model_dir

        # SageMaker will set these env vars automatically; we fallback to args for local runs
        train_path = os.environ.get("SM_CHANNEL_TRAIN", train_path)
        model_dir = os.environ.get("SM_MODEL_DIR", model_dir)

        # Debug info
        print(f"Using train data from: {train_path}")
        print(f"Saving model to: {model_dir}")
        print(f"sys.executable={sys.executable}")
        print(f"python sys.path={sys.path}")
        print(f"pandas version: {pd.__version__}")
        print(
            f"joblib version: {joblib.__version__ if hasattr(joblib, '__version__') else 'unknown'}"
        )
        print(f"sklearn version: {sklearn_version}")

        os.makedirs(model_dir, exist_ok=True)

        # Load data
        df = load_training_data(train_path)
        print(f"Loaded training dataframe shape: {df.shape}")

        # Simple features/target split
        X = df[["size_sqft", "num_rooms", "age_years"]]
        y = df["price"]

        # Train a simple linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Simple training metric for sanity
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        print(f"Training MSE: {mse:.4f}")

        # Save model as model.joblib
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
    except Exception:
        tb = traceback.format_exc()
        print("EXCEPTION in train.py:\n" + tb, file=sys.stderr)
        # try to write a local log
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "train_error.log"), "w") as ef:
                ef.write(tb)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

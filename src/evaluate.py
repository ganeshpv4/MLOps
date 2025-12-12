import argparse
import json
import os
import sys
import tarfile
import traceback
from glob import glob

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
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
    try:
        args = parse_args()

        # Prefer CLI args passed by ProcessingStep (these are explicit).
        # Fall back to SM_* env vars only if args are empty/None.
        model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        test_path = args.test_data or os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
        output_dir = args.output_dir or os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")

        # Debug: show both CLI args and env vars, then list likely locations
        print(
            f"ARG model_dir={args.model_dir} test_data={args.test_data} output_dir={args.output_dir}"
        )
        print(f"ENV SM_MODEL_DIR={os.environ.get('SM_MODEL_DIR')}")
        print(f"ENV SM_CHANNEL_TEST={os.environ.get('SM_CHANNEL_TEST')}")
        print(f"ENV SM_OUTPUT_DATA_DIR={os.environ.get('SM_OUTPUT_DATA_DIR')}")
        print(f"sys.executable={sys.executable}")
        print(f"python sys.path={sys.path}")
        print(f"pandas version: {pd.__version__}")
        print(
            f"joblib version: {joblib.__version__ if hasattr(joblib, '__version__') else 'unknown'}"
        )
        print(f"sklearn version: {sklearn_version}")

        # Ensure output dir exists for logs/metrics
        os.makedirs(output_dir, exist_ok=True)

        # List contents of common model locations to aid debugging
        for p in (model_dir, "/opt/ml/processing/model", "/opt/ml/model"):
            try:
                print(f"LIST {p}: {os.listdir(p)}")
            except Exception as e:
                print(f"Cannot list {p}: {e}")

        # ---- Load model ----
        try:
            model_path = find_or_extract_model(model_dir)
        except FileNotFoundError as e:
            # add directory listing into the exception log for CloudWatch
            tb = f"{e}\nDirectory listings above indicate what is present."
            print(tb, file=sys.stderr)
            # write a helper log file to output dir for easier inspection
            try:
                with open(os.path.join(output_dir, "model_not_found.log"), "w") as ef:
                    ef.write(tb)
            except Exception:
                pass
            raise

        print(f"Resolved model_path={model_path}")
        try:
            print(f"List model_dir contents: {os.listdir(os.path.dirname(model_path))}")
        except Exception:
            pass
        model = joblib.load(model_path)

        # ---- Load test data ----
        df = load_data(test_path)
        print(f"Loaded test dataframe shape: {df.shape}")
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
    except Exception:
        # dump full traceback to stderr and to a file so SageMaker logs show it
        tb = traceback.format_exc()
        print("EXCEPTION in evaluate.py:\n" + tb, file=sys.stderr)
        # also write a log file in output_dir if possible
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "evaluation_error.log"), "w") as ef:
                ef.write(tb)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

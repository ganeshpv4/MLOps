📌 MLOps Mini Project – SageMaker Pipelines + GitHub Actions

This repo contains a sample ML pipeline using Amazon SageMaker Pipelines for training & evaluation, triggered via GitHub Actions CI/CD.

🚀 How it works

Developer pushes code → GitHub Actions runs:

Format check (Black)

Lint (Ruff)

Type check (Mypy)

Unit tests (Pytest)

If CI passes → SageMaker Pipeline is automatically upserted & executed.

Pipeline steps:

TrainModel → trains with src/train.py

EvaluateModel → evaluates with src/evaluate.py and stores metrics.json in S3

👩‍💻 Developer Setup

python -m pip install -r requirements.txt

python -m pip install pre-commit

python -m pre_commit install


Workflow:

git add .

git commit -m "msg"   # pre-commit auto-formats & lints

git commit -m "msg"   # commit again if first try modified files

git push              # CI + SageMaker pipeline run

🧪 Run tests locally

python -m pytest -q

▶️ Manually run SageMaker pipeline (optional)

python -m pipelines.run_pipeline \
  --region <region> \
  --role-arn <role> \
  --default-bucket <s3-bucket> \
  --input-data-uri <s3-input-csv>

Run locally

Recommended setup

Create a venv and install minimal deps

python -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install pandas scikit-learn joblib

# Optionally run full requirements if you want CI/tools: pip install -r requirements.txt

Train locally

python -m src.train --train-data data/housing.csv --model-dir model

Evaluate locally

python -m src.evaluate --model-dir model --test-data data/housing.csv --output-dir eval

🔐 Required GitHub Secrets

Secrets

AWS_ACCESS_KEY_ID	AWS

AWS_SECRET_ACCESS_KEY

SAGEMAKER_ROLE_ARN

⭐ Notes

Always commit formatted code (pre-commit takes care of this).

CI must be green for SageMaker pipeline to run.

Shut down SageMaker Studio apps when not in use to avoid cost.
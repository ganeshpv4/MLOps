import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def get_pipeline(
    region: str,
    role_arn: str,
    default_bucket: str,
    pipeline_name: str = "house-price-demo-pipeline",
) -> Pipeline:
    """
    Build a simple SageMaker Pipeline:
      1. Train model using src/train.py
      2. Evaluate model using src/evaluate.py
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)

    # -------- Pipeline parameters (can be overridden at runtime) --------
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{default_bucket}/house-prices/housing.csv",
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge",
    )

    # NEW: processing instance type parameter (use a smaller instance by default)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.t3.medium",
    )

    # Where evaluation outputs go
    evaluation_output_prefix = f"s3://{default_bucket}/house-prices/evaluation"

    # -------- Training step (SKLearn Estimator) --------
    sklearn_version = "1.2-1"  # adjust if needed
    py_version = "py3"

    estimator = SKLearn(
        entry_point="train.py",  # relative to source_dir
        source_dir="src",  # our src folder in the repo
        role=role_arn,
        instance_count=1,
        instance_type=training_instance_type,
        framework_version=sklearn_version,
        py_version=py_version,
        sagemaker_session=sagemaker_session,
        base_job_name="house-price-train",
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=input_data_uri,
                content_type="text/csv",
            )
        },
    )

    # -------- Evaluation step (SKLearnProcessor) --------
    processor = SKLearnProcessor(
        framework_version=sklearn_version,
        role=role_arn,
        instance_type=processing_instance_type,  # <-- changed from hardcoded m5.xlarge
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name="house-price-eval",
    )

    # We want to read metrics.json from the evaluation output
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="metrics.json",
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=processor,
        inputs=[
            # Model artifacts from training job
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            # Test data (for simplicity, use same InputDataUri)
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/test",
                input_name="test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=evaluation_output_prefix,
            ),
        ],
        code="src/evaluate.py",
        job_arguments=[
            "--model-dir",
            "/opt/ml/processing/model",
            "--test-data",
            "/opt/ml/processing/test",
            "--output-dir",
            "/opt/ml/processing/output",
        ],
        property_files=[evaluation_report],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data_uri, training_instance_type, processing_instance_type],
        steps=[train_step, eval_step],
        sagemaker_session=sagemaker_session,
    )

    return pipeline

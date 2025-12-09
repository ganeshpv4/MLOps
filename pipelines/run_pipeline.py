# pipelines/run_pipeline.py

import argparse

import boto3

from pipelines.pipeline import get_pipeline


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--region", required=True, help="AWS region, e.g. ap-south-1")
    parser.add_argument(
        "--role-arn",
        required=True,
        help="SageMaker execution role ARN",
    )
    parser.add_argument(
        "--pipeline-name",
        default="house-price-demo-pipeline",
        help="Name of the SageMaker pipeline",
    )
    parser.add_argument(
        "--default-bucket",
        required=False,
        help="S3 bucket for default pipeline artifacts. "
        "If not set, will use SageMaker default bucket.",
    )
    parser.add_argument(
        "--input-data-uri",
        required=False,
        help="S3 URI of training/eval data; overrides pipeline parameter",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    import sagemaker

    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)

    if args.default_bucket:
        default_bucket = args.default_bucket
    else:
        default_bucket = sagemaker_session.default_bucket()

    pipeline = get_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        default_bucket=default_bucket,
        pipeline_name=args.pipeline_name,
    )

    # Create or update pipeline definition
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Upserted pipeline: {args.pipeline_name}")

    # Optional parameters override
    runtime_parameters = {}
    if args.input_data_uri:
        runtime_parameters["InputDataUri"] = args.input_data_uri

    execution = pipeline.start(parameters=runtime_parameters or None)

    print("Started pipeline execution:")
    print(f"  ARN: {execution.arn}")


if __name__ == "__main__":
    main()

# MLOps

MLOps/
├── src/
│   ├── train.py          # training script (used by SageMaker)
│   └── evaluate.py       # evaluation script (used in pipeline)
├── pipelines/
│   └── pipeline.py       # SageMaker pipeline definition
├── data/                 # local data for quick testing (optional)
│   └── housing.csv
├── requirements.txt
└── README.md

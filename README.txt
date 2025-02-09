========================================
Assignment 1 - Machine Learning Experiments

This repository contains the implementation and experiments for Assignment 1, comparing non-parametric learning, parametric learning, and deep learning using two datasets:

1. Customer Personality Dataset (Task 1)

2. Spotify 2023 Dataset (Task 2)

Repository Structure

Assignment-1/
│── configs/               --> Configuration files for models and experiments (YAML format)
│── datasets/              --> Preprocessed dataset files (No need to rerun data pipeline)
│── model_checkpoints/     --> Saved model checkpoints from previous runs
│── notebooks/             --> Jupyter notebooks for additional analysis or visualization
│── results/               --> Output results including accuracy, classification reports, and F1 scores
│── src/                   --> Experiment scripts for model training and evaluation
│── README.txt             --> (This file) Assignment 1 README
│── environment.yaml       --> Conda environment dependencies for reproducing experiments

Experiment Files

The primary experiment scripts are stored in the src/ directory:

1. mkt_campaign_experiment.ipynb  --> Experiment for Customer Personality Dataset (Task 1)

2. spotify_experiment.ipynb  --> Experiment for Spotify 2023 Dataset (Task 2)

Important Notes

1. No need to rerun the data pipeline in the first cell of the experiment notebooks. The required datasets are already stored in the datasets/ directory.

2. To reproduce the results, 
    1. Update the corresponding model configuration in the YAML files inside the configs/ folder if different from the configuration mentioned in the report.

    2. To view metrics (accuracy, classification report, F1 score): Open the corresponding experiment notebook (experiment.ipynb). Rerun the cells to generate the evaluation results.

Environment Setup

To install dependencies, create a new Conda environment using:

conda env create -f environment.yaml
conda activate <your_env_name>

For any issues or clarifications, please refer to the Assignment 1 Report.

Read Only Project Report: https://www.overleaf.com/read/jqvtccffpcym#0df13d


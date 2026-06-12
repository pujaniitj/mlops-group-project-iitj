End-to-End MLOps Pipeline — Sentiment Analysis on IMDB

IIT Jodhpur | PGD AI Program | MLOps (CSL7040) | Group 8


Team Members

MemberFull NameRoll NoGitHub UsernameTasksMember 1 (Admin)Pujan ChakrabortyG25AIT2076pujaniitjTasks 1, 2, 3, 4, 8Member 2Mannu Singh[TO BE ADDED]manuiitjTask 5Member 3Rahul SharmaG25AIT2144g25ait2144Task 6Member 4Sai Chaitanya[TO BE ADDED]g25ait2143-specTask 7


Live Project Links

All links are publicly accessible.

ResourceURLGitHub Repositoryhttps://github.com/pujaniitj/mlops-group-project-iitjHugging Face Modelhttps://huggingface.co/Pujaniitj/MLOPS_GROUP_PROJECTDocker Hub Imagehttps://hub.docker.com/r/g25ait2144/mlops-group-projectW&B Dashboardhttps://wandb.ai/pujaniitj-iit-jodpur/MLops_group_8Kaggle Notebook V1https://www.kaggle.com/code/pujaniitj/mlops-group-8-imdb-v1Kaggle Notebook V2https://www.kaggle.com/code/pujaniitj/mlops-group-8-imdb-v2

Project Overview

This project demonstrates a complete end-to-end MLOps pipeline for binary sentiment classification on IMDB movie reviews. The pipeline covers every stage from raw data ingestion and model fine-tuning, through experiment tracking, to containerized inference and automated CI/CD workflows via GitHub Actions.

PropertyValueTaskBinary Sentiment Classification (Positive / Negative)Datasetstanfordnlp/imdb — 50,000 reviews from Hugging Face HubBase Modeldistilbert-base-uncased (67M parameters, 268 MB)Production Modelrun-v1 (learning rate = 3e-5)Training PlatformKaggle Notebooks (NVIDIA Tesla T4 x2 GPU)Experiment TrackingWeights and Biases — project: MLops_group_8


## GitHub Actions CI/CD

Two automated workflows are configured in .github/workflows/:

### CI Workflow — Lint and Validate

Triggers automatically on every push or pull request to the develop branch.

- Checks out the code
- Sets up Python 3.10
- Runs flake8 linter on src/ with max line length 120
- Fails the build if any critical code errors are found

*Status badge:*

![CI](https://github.com/pujaniitj/mlops-group-project-iitj/actions/workflows/ci.yml/badge.svg?branch=develop)

### Inference Workflow — Run Sentiment Analysis

Triggered manually via GitHub Actions UI (workflow_dispatch).

Accepts custom text input and runs the DistilBERT sentiment classifier:

1. Go to *Actions* tab
2. Click *Inference - Run Sentiment Analysis*
3. Click *Run workflow*
4. Enter text to classify
5. Click the green *Run workflow* button
6. Wait ~1 minute for the result

*Sample output:*


========================================
  Sentiment:  negative
  Confidence: 0.9963
========================================


### Workflow files

| File | Trigger | Purpose |
|---|---|---|
| .github/workflows/ci.yml | Push / PR to develop | Lint with flake8 |
| .github/workflows/inference.yml | Manual (workflow_dispatch) | Run sentiment inference |

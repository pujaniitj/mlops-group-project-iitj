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

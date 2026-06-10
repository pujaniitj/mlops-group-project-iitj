## Docker Container

The inference script is packaged as a Docker container and available on Docker Hub:

🐳 **[g25ait2144/mlops-group-project](https://hub.docker.com/r/g25ait2144/mlops-group-project)**

### Pull and run

bash
docker pull g25ait2144/mlops-group-project:latest


bash
docker run --rm \
  -e INPUT_TEXT="This movie was absolutely fantastic!" \
  g25ait2144/mlops-group-project:latest


### Sample output


Model: pujaniitj/MLOPS_GROUP_PROJECT
Input: Worst film I have ever watched
Loading model from Hugging Face Hub...
Running inference...

========================================
  Sentiment:  negative
  Confidence: 0.9963
========================================


### Environment variables

| Variable | Default | Description |
|---|---|---|
| INPUT_TEXT | Sample positive review | Text to classify |
| HF_MODEL_NAME | pujaniitj/MLOPS_GROUP_PROJECT | Hugging Face model to load |
| HF_TOKEN | None | Optional — only for private HF repos |

### Build locally

bash
git clone https://github.com/pujaniitj/mlops-group-project-iitj.git
cd mlops-group-project-iitj
docker build -t mlops-group-project:latest .
docker run --rm -e INPUT_TEXT="Your text here" mlops-group-project:latest

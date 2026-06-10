FROM python:3.10-slim

WORKDIR /app

ARG HF_MODEL_NAME=Pujaniitj/MLOPS_GROUP_PROJECT
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/inference.py"]
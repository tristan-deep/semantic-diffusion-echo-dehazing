FROM zeahub/all:v0.0.4

RUN pip install --no-cache-dir gradio tyro optuna

RUN pip install --no-cache-dir --no-deps pytorch_fid

RUN pip install --no-cache-dir -U keras

RUN apt-get update && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

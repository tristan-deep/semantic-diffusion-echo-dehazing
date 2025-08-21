FROM zeahub/all:v0.0.4

RUN pip install --no-cache-dir tyro optuna

RUN pip install --no-cache-dir --no-deps pytorch_fid

RUN pip install --no-cache-dir -U keras

WORKDIR /workspace

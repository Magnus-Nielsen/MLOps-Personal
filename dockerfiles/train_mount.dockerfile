# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY data/ data/
COPY data/ data/
COPY src/corrupted_mnist/ src/corrupted_mnist/
COPY src/corrupted_mnist/ src/corrupted_mnist/

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/corrupted_mnist/train.py"]
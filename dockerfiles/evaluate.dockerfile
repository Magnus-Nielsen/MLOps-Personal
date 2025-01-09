# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/corrupted_mnist/ src/corrupted_mnist/
COPY reports/ reports/
COPY models/ models/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/corrupted_mnist/evaluate.py"]

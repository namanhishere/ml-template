# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /workspace

COPY pyproject.toml ./
RUN uv pip install --system -e ".[dev]"

COPY . .

RUN mkdir -p checkpoints outputs logs

ENTRYPOINT ["python", "scripts/train.py"]

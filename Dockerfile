# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps used by some Python wheels / runtime helpers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install.
COPY . /app
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

EXPOSE 9000 8088

CMD ["llama-conductor", "serve", "--host", "0.0.0.0", "--port", "9000"]

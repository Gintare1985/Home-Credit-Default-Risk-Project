# syntax=docker/dockerfile:1.4
FROM --platform=linux/amd64 python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.0.0
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /ml_app

COPY pyproject.toml poetry.lock* /ml_app/

RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi --without dev

COPY . /ml_app

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn ml_app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
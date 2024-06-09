FROM python:3.10-slim
WORKDIR /usr/src/app

# Handle dependency installation
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-dev
COPY references_searcher ./references_searcher
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Copy the rest of the files
COPY model_weights ./model_weights
COPY config ./config
COPY scripts ./scripts
COPY web_app ./web_app

EXPOSE 8000
RUN chmod +x ./scripts/docker_entrypoint.sh
CMD ["./scripts/docker_entrypoint.sh"]
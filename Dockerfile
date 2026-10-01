FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/ragbench-models
WORKDIR /app

# libgomp also supports the optional FAISS/PyTorch CPU dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home ragbench \
    && mkdir /data && chown ragbench:ragbench /data
COPY pyproject.toml README.md LICENSE ./
COPY ragbench ./ragbench
ARG EXTRAS=api
RUN pip install --no-cache-dir ".[${EXTRAS}]"
COPY configs ./configs
COPY datasets ./datasets
COPY benchmarks ./benchmarks
USER 10001
ENTRYPOINT ["ragbench"]
CMD ["evaluate", "--config", "configs/baseline.yaml"]

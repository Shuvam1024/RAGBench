FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    HF_HOME=/tmp/ragstat-models \
    PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv
WORKDIR /app

# libgomp also supports the optional FAISS/PyTorch CPU dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home ragstat \
    && mkdir /data && chown ragstat:ragstat /data
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY ragstat ./ragstat
ARG EXTRAS=api
RUN extras=""; \
    for extra in $(printf '%s' "$EXTRAS" | tr ',' ' '); do extras="$extras --extra $extra"; done; \
    uv sync --frozen --no-dev $extras \
    && chmod -R a+rX /app/.venv
COPY configs ./configs
COPY datasets ./datasets
COPY benchmarks ./benchmarks
USER 10001
ENTRYPOINT ["ragstat"]
CMD ["evaluate", "--config", "configs/baseline.yaml"]

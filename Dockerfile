FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 postgresql-client \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ops --no-install-project
COPY msynd/ ./msynd/
COPY data/params.json data/segment_overrides.csv data/holidays.csv data/static_sources.json ./data/
RUN python -c "import lightgbm, onnxruntime, numba"

CMD ["python", "-m", "msynd.engine"]

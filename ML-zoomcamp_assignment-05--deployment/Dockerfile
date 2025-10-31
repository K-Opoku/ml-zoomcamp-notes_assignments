
FROM ghcr.io/astral-sh/uv:latest as uv-builder

FROM python:3.13.5-slim-bookworm

COPY --from=uv-builder /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"


COPY pyproject.toml uv.lock .python-version 

RUN uv sync --locked


COPY score.py .
COPY pipeline_v1.bin .

EXPOSE 9696

ENTRYPOINT ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "9696"]

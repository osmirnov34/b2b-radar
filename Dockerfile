FROM python:3.12.3-slim

WORKDIR /app

RUN useradd -m -u 1000 appuser

RUN pip install --upgrade pip

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "app.py"]

FROM python:3.12.3-slim

WORKDIR /app

RUN useradd -m -u 1000 appuser

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

CMD ["python", "app.py"]

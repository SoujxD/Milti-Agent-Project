FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_SENTENCE_TRANSFORMERS=false \
    MPLCONFIGDIR=/tmp/matplotlib-cache

WORKDIR /app

COPY requirements.fly.txt .
RUN pip install --upgrade pip && pip install -r requirements.fly.txt
RUN mkdir -p /tmp/matplotlib-cache

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

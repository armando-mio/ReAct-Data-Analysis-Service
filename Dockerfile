# Production Dockerfile for ReAct Data Analysis Service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for process management and building
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root application user for secure sandbox execution
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/storage/artifacts /app/storage/uploads && \
    chown -R appuser:appuser /app

# Copy application codebase
COPY --chown=appuser:appuser data_agent /app/data_agent

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "data_agent.adapters.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

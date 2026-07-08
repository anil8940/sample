# Build stage
FROM python:3.14-slim AS builder

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy only dependency files first (leverage layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv into a virtual environment
RUN uv sync --no-dev --frozen

# Runtime stage
FROM python:3.14-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app.py config.py models.py routes.py main.py ./
COPY core/ ./core/
COPY static/ ./static/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/docs')"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

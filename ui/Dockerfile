# UI Dockerfile for Compass Streamlit Frontend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy minimal UI requirements
COPY ui/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy UI source code
COPY ui ./ui

# Copy Streamlit configuration
COPY .streamlit ./.streamlit

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

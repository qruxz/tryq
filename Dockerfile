# Use the Python 3 Alpine official image
FROM python:3-alpine

# Set working directory
WORKDIR /app

# Copy local code into the container
COPY . .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, but good practice)
EXPOSE 8000

# Run the web service on container startup with Gunicorn + Uvicorn workers
CMD ["gunicorn", "app:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

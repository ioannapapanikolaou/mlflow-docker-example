# Use Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies and source files
COPY requirements.txt requirements.txt
COPY app/train.py train.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run MLflow script
CMD ["python", "train.py"]


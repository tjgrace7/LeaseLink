# Use an official Python image that allows apt
FROM python:3.11-slim

# Install tesseract + poppler-utils
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Start your app (adjust path/module if needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]

# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies (for tesseract support if needed)
RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=10000"]


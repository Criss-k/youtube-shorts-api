FROM python:3.12-slim

# Set timezone to Riga (Latvia)
ENV TZ=Europe/Riga
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output

# Ensure fonts directory exists and is accessible
RUN mkdir -p fonts/Lexend/static && \
    echo "Note: Make sure to add your fonts to the fonts directory before building the image"

# Expose the port
EXPOSE 8080

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 
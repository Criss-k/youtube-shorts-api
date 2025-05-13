# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 8080

# Create and set the working directory
WORKDIR $APP_HOME

# --- Install FFmpeg ---
# Needed by libraries like moviepy or ffmpeg-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at $APP_HOME
COPY . .

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run the application
# Use gunicorn for production deployment in Cloud Run for better performance/concurrency
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
# Or use uvicorn directly (simpler, might be sufficient depending on load)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

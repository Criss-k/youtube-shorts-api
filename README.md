# YouTube Shorts Generation API

An API service for generating YouTube Shorts style videos that can be deployed to Google Cloud Run.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Docker
- A Google Cloud Platform project with Cloud Run and Cloud Storage APIs enabled
- Proper permissions to deploy to Cloud Run and access Cloud Storage

## Environment Variables

The service requires the following environment variables:

- `PORT` (provided by Cloud Run automatically)
- `OUTPUT_BUCKET_NAME` - Google Cloud Storage bucket name for storing output videos
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account credentials (managed by Cloud Run if using IAM)

## Font Files

The service uses font files for text rendering. Make sure to include these files in the `fonts/Lexend/static/` directory before building the Docker image.

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application locally:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

## Deployment to Cloud Run

1. Build the Docker image:
   ```
   docker build -t gcr.io/[YOUR-PROJECT-ID]/youtube-shorts-api .
   ```

2. Push the image to Google Container Registry:
   ```
   docker push gcr.io/[YOUR-PROJECT-ID]/youtube-shorts-api
   ```

3. Deploy to Cloud Run:
   ```
   gcloud run deploy youtube-shorts-api \
     --image gcr.io/[YOUR-PROJECT-ID]/youtube-shorts-api \
     --platform managed \
     --region [REGION] \
     --set-env-vars "OUTPUT_BUCKET_NAME=[YOUR-BUCKET-NAME]" \
     --allow-unauthenticated
   ```

4. Alternatively, use the Cloud Run button in the Google Cloud Console to deploy directly from the container registry.

## Service Configuration

- Memory: Recommend at least 2GB
- CPU: Recommend at least 1 CPU
- Maximum request timeout: 15 minutes (for video processing)
- Concurrency: 1 request per instance (video processing is CPU and memory intensive)

## API Usage

See the API documentation by accessing the `/docs` endpoint after deployment.

## Monitoring and Logging

Cloud Run provides built-in logging and monitoring. You can view logs in the Google Cloud Console or using the `gcloud` command:

```
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=youtube-shorts-api"
``` 
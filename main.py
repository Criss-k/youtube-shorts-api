import os
import json
import tempfile
import logging
import random
import numpy as np
from PIL import Image
import math
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Tuple
import requests
from google.cloud import storage
from moviepy import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, \
                         TextClip, CompositeVideoClip, VideoFileClip
from moviepy.audio.fx import MultiplyVolume
import cv2

# --- Configuration ---
# Cloud Run dynamically sets the PORT environment variable.
PORT = int(os.environ.get("PORT", 8080))
OUTPUT_BUCKET_NAME = "n8n-bucket-yt" # Bucket to store final videos
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
IMAGE_DURATION_S = 6
TARGET_ASPECT_RATIO = VIDEO_WIDTH / VIDEO_HEIGHT
BG_MUSIC_VOLUME = 0.2 # Adjust background music volume (0.0 to 1.0)
TEXT_FONT = 'Arial' # Ensure this font is available in the container or provide font file
TEXT_FONT_SIZE = 60
TEXT_COLOR = 'white'
TEXT_STROKE_COLOR = 'black'
TEXT_STROKE_WIDTH = 2
TEXT_POSITION = ('center', 'bottom') # Position relative to video frame
TEXT_MARGIN_BOTTOM = 50 # Pixels from bottom if position is bottom

# --- Pydantic Models for Request Validation ---
class TranscriptItem(BaseModel):
    words: str
    start: float
    end: float

class VideoData(BaseModel):
    background_music_url: HttpUrl
    image_urls: List[HttpUrl]
    voice_url: HttpUrl
    transcripts: List[TranscriptItem] # Changed from str to List[TranscriptItem]

class VideoRequest(BaseModel):
    type: str = Field(..., pattern="^ImageVideo$") # Only allow "ImageVideo"
    data: VideoData

# --- FastAPI App ---
app = FastAPI()

# --- Custom Exception Handler for Validation Errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request: {request.method} {request.url}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    # You can customize the response, but FastAPI's default is usually good.
    # Return a generic error message or include exc.errors() if you want to expose details.
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body} # Pydantic v2 uses exc.body
    )

# --- Helper Functions ---

def download_file(url: str, destination_folder: str) -> str:
    """Downloads a file from a URL to a local path."""
    try:
        response = requests.get(url, stream=True, timeout=60) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes
        # Extract filename or generate one
        filename = url.split('/')[-1] or f"downloaded_{os.urandom(4).hex()}"
        local_path = os.path.join(destination_folder, filename)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {url} to {local_path}")
        return local_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download asset: {url}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during download.")

def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str) -> str:
    """Uploads a file to Google Cloud Storage and returns the public URL."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Consider making it publically readable upon upload, or generate signed URL
        blob.upload_from_filename(local_path)
        # Construct the public URL (ensure bucket/object ACLs allow public access)
        # This format assumes public access is enabled. For private, use signed URLs.
        public_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
        logger.info(f"Successfully uploaded {local_path} to {public_url}")
        return public_url
    except Exception as e: # Catch specific GCS exceptions if needed
        logger.error(f"Failed to upload {local_path} to GCS bucket {bucket_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload final video to storage.")

def create_ken_burns_clip(image_path: str, duration: float, target_size: Tuple[int, int]) -> ImageClip:
    """Creates a Ken Burns effect (pan/zoom) for an image clip."""
    try:
        # Load the image using PIL first to get dimensions
        pil_img = Image.open(image_path)
        img_w, img_h = pil_img.size
        target_w, target_h = target_size
        
        # Calculate scaling factors to fill the target size while maintaining aspect ratio
        scale_w = target_w / img_w
        scale_h = target_h / img_h
        scale = max(scale_w, scale_h)
        
        # Calculate initial and final zoom values
        zoom_start = scale
        zoom_end = scale * random.uniform(1.1, 1.3)  # Zoom in 10-30%
        
        # Decide on pan direction (random)
        # 0: zoom center, 1: left to right, 2: right to left, 3: top to bottom, 4: bottom to top
        pan_type = random.randint(0, 4)
        
        # For MoviePy 2.1.2, we'll use a simpler approach by creating frames in advance
        # Create a sequence of frames for the Ken Burns effect
        frames = []
        num_frames = int(duration * 24)  # Assuming 24 fps
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            current_zoom = zoom_start + (zoom_end - zoom_start) * progress
            
            # Load and convert image
            img = np.array(pil_img)
            
            # Calculate new dimensions
            new_h = int(img_h * current_zoom)
            new_w = int(img_w * current_zoom)
            
            # Resize using OpenCV
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate crop position based on pan type
            if pan_type == 0:  # Center zoom
                y = (new_h - target_h) // 2
                x = (new_w - target_w) // 2
            elif pan_type == 1:  # Left to right
                y = (new_h - target_h) // 2
                x = int((new_w - target_w) * progress)
            elif pan_type == 2:  # Right to left
                y = (new_h - target_h) // 2
                x = int((new_w - target_w) * (1 - progress))
            elif pan_type == 3:  # Top to bottom
                x = (new_w - target_w) // 2
                y = int((new_h - target_h) * progress)
            else:  # Bottom to top
                x = (new_w - target_w) // 2
                y = int((new_h - target_h) * (1 - progress))
            
            # Ensure we don't go out of bounds
            x = min(max(0, x), new_w - target_w)
            y = min(max(0, y), new_h - target_h)
            
            # Crop to target size
            cropped = resized[y:y+target_h, x:x+target_w]
            
            # Convert back to RGB if needed
            if cropped.shape[2] == 4:  # If RGBA
                cropped = cv2.cvtColor(cropped, cv2.COLOR_RGBA2RGB)
            
            frames.append(cropped)
        
        # Create a VideoFileClip from frames
        clip = ImageClip(frames[0], duration=duration)  # Start with first frame
        return clip
    except Exception as e:
        logger.error(f"Error creating Ken Burns clip for {image_path}: {e}")
        # Create a static clip as fallback
        try:
            # Load image with OpenCV and resize
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate scaling to maintain aspect ratio
            h, w = img.shape[:2]
            scale = max(target_size[0]/w, target_size[1]/h)
            new_size = (int(w*scale), int(h*scale))
            
            # Resize image
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Center crop
            y = (new_size[1] - target_size[1]) // 2
            x = (new_size[0] - target_size[0]) // 2
            cropped = resized[y:y+target_size[1], x:x+target_size[0]]
            
            # Create static clip directly with duration parameter
            return ImageClip(cropped, duration=duration)
        except Exception as inner_e:
            logger.error(f"Failed to create fallback static clip: {inner_e}")
            # Create a blank clip as last resort
            blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            return ImageClip(blank, duration=duration)


def process_video_task(request_data: VideoData):
    """
    Downloads assets, generates video using moviepy, uploads result.
    """
    logger.info("Starting video processing task...")
    output_video_filename = f"output_{os.urandom(8).hex()}.mp4"
    final_video_url = None # Initialize

    with tempfile.TemporaryDirectory() as tmpdir:
        output_video_path = os.path.join(tmpdir, output_video_filename)
        try:
            logger.info(f"Using temporary directory: {tmpdir}")

            # 1. Download Assets
            logger.info("Downloading assets...")
            voice_path = download_file(str(request_data.voice_url), tmpdir)
            bg_music_path = download_file(str(request_data.background_music_url), tmpdir)
            image_paths = [download_file(str(url), tmpdir) for url in request_data.image_urls]

            # Transcripts are now directly parsed Pydantic models
            transcripts_list = request_data.transcripts # Directly use the parsed list
            logger.info(f"Received {len(transcripts_list)} transcript items.")

            # --- Basic Validation ---
            if not image_paths:
                raise ValueError("No image URLs provided or failed to download images.")

            # 2. Load Audio
            logging.info("Loading audio files...")
            try:
                voice_audio = AudioFileClip(voice_path)
                # Load background music
                bg_music = AudioFileClip(bg_music_path)
                
                # Adjust background music duration first
                if bg_music.duration > voice_audio.duration:
                    bg_music = bg_music.subclip(0, voice_audio.duration)
                elif bg_music.duration < voice_audio.duration:
                    # Loop background music if shorter than video
                    bg_music = bg_music.loop(duration=voice_audio.duration)
                
                # Store final duration
                video_duration = voice_audio.duration
                logger.info(f"Voice audio duration: {video_duration:.2f} seconds")
                
                # Apply volume adjustment directly to the audio array
                bg_music = bg_music.volumex(BG_MUSIC_VOLUME)
            except Exception as e:
                logging.error(f"Error loading audio file: {e}")
                raise

            # --- Create Image Sequence Video ---
            logger.info("Creating image sequence with Ken Burns effect...")
            clips = []
            current_time = 0
            image_index = 0
            while current_time < video_duration:
                img_path = image_paths[image_index % len(image_paths)] # Loop images
                clip_duration = min(IMAGE_DURATION_S, video_duration - current_time)
                if clip_duration <= 0: break # Avoid zero duration clips

                try:
                    kb_clip = create_ken_burns_clip(img_path, clip_duration, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    kb_clip = kb_clip.with_start(current_time) # Not strictly needed for concatenation but good practice
                    clips.append(kb_clip)
                    logger.debug(f"Created clip for image {image_index % len(image_paths)}: start={current_time:.2f}, duration={clip_duration:.2f}")
                except Exception as e:
                    logger.error(f"Error creating Ken Burns clip for {img_path}: {e}. Skipping image.")
                    # Add a blank clip or handle differently?
                    # For now, just skip, might shorten video.

                current_time += clip_duration
                image_index += 1

            if not clips:
                 raise ValueError("Failed to create any video clips from images.")

            final_image_video = concatenate_videoclips(clips).with_duration(video_duration)
            logger.info(f"Concatenated image video duration: {final_image_video.duration:.2f} seconds")

            # --- Create Text Overlays ---
            logger.info("Creating text overlays...")
            text_clips = []
            for item in transcripts_list:
                start_time = item.start
                end_time = item.end
                text_duration = end_time - start_time
                if text_duration <= 0:
                    logger.warning(f"Skipping transcript item with zero/negative duration: '{item.words}'")
                    continue

                # Create text clip
                try:
                    txt_clip = TextClip(
                        item.words,
                        fontsize=TEXT_FONT_SIZE,
                        color=TEXT_COLOR,
                        stroke_color=TEXT_STROKE_COLOR,
                        stroke_width=TEXT_STROKE_WIDTH,
                        size=(VIDEO_WIDTH * 0.9, None), # Limit width to 90% of video
                        method='caption' # Auto-wrap text
                    )
                    # Set font separately to avoid multiple values error
                    txt_clip = txt_clip.with_font(TEXT_FONT)
                    
                    # Calculate position
                    pos_x, pos_y = TEXT_POSITION
                    clip_w, clip_h = txt_clip.size
                    if pos_y == 'bottom':
                         final_pos_y = VIDEO_HEIGHT - clip_h - TEXT_MARGIN_BOTTOM
                    elif pos_y == 'center':
                         final_pos_y = (VIDEO_HEIGHT - clip_h) / 2
                    else: # Assume top or numeric
                         final_pos_y = pos_y

                    if pos_x == 'center':
                         final_pos_x = (VIDEO_WIDTH - clip_w) / 2
                    else: # Assume left/right or numeric
                        final_pos_x = pos_x

                    txt_clip = txt_clip.with_position((final_pos_x, final_pos_y))
                    txt_clip = txt_clip.with_start(start_time)
                    txt_clip = txt_clip.with_duration(text_duration)
                    text_clips.append(txt_clip)
                    logger.debug(f"Created text clip: '{item.words}' start={start_time:.2f} end={end_time:.2f}")
                except Exception as e:
                    logger.error(f"Error creating text clip for '{item.words}': {e}")
                    # Decide whether to fail or continue without this text

            # --- Combine Audio --- #
            logger.info("Combining audio tracks...")
            final_audio = CompositeAudioClip([voice_audio, bg_music])
            # Ensure final audio has the exact duration
            final_audio = final_audio.with_duration(video_duration)

            # --- Composite Video --- #
            logger.info("Compositing final video...")
            # Combine image sequence video with all text clips
            video_with_text = CompositeVideoClip([final_image_video] + text_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            video_with_text = video_with_text.with_duration(video_duration)
            # Set the combined audio to the final video
            final_clip = video_with_text.with_audio(final_audio)

            # 4. Write Video File
            logger.info(f"Writing final video to {output_video_path}...")
            # Use appropriate codecs for web compatibility (H.264/AAC are common)
            # threads=4 can speed up encoding, adjust based on Cloud Run CPU
            # logger='bar' provides progress
            final_clip.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(tmpdir, 'temp-audio.m4a'), # Ensure temp file is in tmpdir
                remove_temp=True,
                fps=24,
                preset='medium', # 'medium' is a balance, 'fast' or 'ultrafast' for speed
                threads=4,
                logger='bar' # or None to disable progress bar logging
            )
            logger.info("Video file written successfully.")

            # 5. Upload Result
            logger.info(f"Uploading result: {output_video_path}")
            final_video_url = upload_to_gcs(output_video_path, OUTPUT_BUCKET_NAME, output_video_filename)
            logger.info(f"Video processing complete. Final URL: {final_video_url}")
            # Consider adding success notification here

        except (HTTPException, ValueError, requests.exceptions.RequestException) as e:
            logger.error(f"Error during video processing task: {e}")
            # Potentially log specific details or notify about failure
        except Exception as e:
            logger.exception(f"Unhandled exception during video processing task: {e}")
            # Log the full traceback for unexpected errors
        finally:
            # Cleanup moviepy's temporary files if any linger (though it usually cleans up)
            # Close clips if necessary (moviepy usually handles this)
            try:
                if 'voice_audio' in locals() and voice_audio: voice_audio.close()
                if 'bg_music' in locals() and bg_music: bg_music.close()
                if 'final_image_video' in locals() and final_image_video: final_image_video.close()
                if 'final_audio' in locals() and final_audio: final_audio.close()
                if 'video_with_text' in locals() and video_with_text: video_with_text.close()
                if 'final_clip' in locals() and final_clip: final_clip.close()
                # Also close any text clips
                if 'text_clips' in locals() and text_clips:
                    for clip in text_clips:
                        if clip: clip.close()
            except Exception as e:
                logger.warning(f"Error during clip cleanup: {e}")
                
            logger.info(f"Finished processing. Cleaned up temporary directory: {tmpdir}")
            # Optional: Update a status in a DB or send notification about success/failure
            if final_video_url:
                logger.info(f"Task succeeded. Video available at: {final_video_url}")
            else:
                logger.error("Task failed to produce a video URL.")


# --- API Endpoint ---
@app.post("/")
async def create_short_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    API endpoint to trigger YouTube Shorts video generation.
    Accepts POST requests with asset URLs and transcript data.
    Initiates a background task for processing.
    """
    logger.info(f"Received request for type: {request.type}")
    if request.type == "ImageVideo":
        # Add the processing to background tasks
        background_tasks.add_task(process_video_task, request.data)
        logger.info("Video generation task added to background.")
        # Return an immediate response acknowledging the request
        return {"message": "Video generation started successfully. Processing occurs in the background."}
    else:
        # Although Pydantic validation catches this, adding belt-and-suspenders
        logger.warning(f"Received unsupported request type: {request.type}")
        raise HTTPException(status_code=400, detail=f"Unsupported type: {request.type}. Only 'ImageVideo' is supported.")

# --- Uvicorn Runner (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server locally on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

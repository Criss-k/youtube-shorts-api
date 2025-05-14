import os
import json
import tempfile
import logging
import random
import numpy as np
from PIL import Image
import math
import shutil
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Tuple, Optional
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
LOCAL_OUTPUT_DIR = "./output" # Local directory to save copies of output files
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

class KenBurnsTestRequest(BaseModel):
    image_urls: List[HttpUrl]
    duration_per_image: Optional[float] = 6.0

class AnimationTestRequest(BaseModel):
    image_urls: List[HttpUrl]
    duration_per_image: Optional[float] = 6.0
    effect_type: str = "ken_burns"  # Default to ken_burns, will support more in the future
    effect_params: Optional[dict] = None  # Optional parameters specific to each effect

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
    """
    Creates a Ken Burns effect (pan/zoom) for an image clip.
    
    This implementation handles images of different aspect ratios by properly
    preprocessing them and ensuring the Ken Burns effect works correctly regardless
    of image orientation (portrait, landscape, or square).
    """
    try:
        target_w, target_h = target_size
        target_aspect = target_w / target_h
        
        # Load the image using PIL first to get dimensions
        pil_img = Image.open(image_path)
        img_w, img_h = pil_img.size
        img_aspect = img_w / img_h
        
        # Convert PIL image to numpy array
        img_array = np.array(pil_img)
        
        # Handle RGBA images
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Determine zoom and pan parameters
        is_portrait = img_aspect < target_aspect
        is_landscape = img_aspect > target_aspect
        
        # Determine the minimum zoom required to fill the target frame
        if is_portrait:  # Tall/narrow image
            min_zoom = target_w / img_w  # Zoom to match width
        else:  # Wide/landscape image
            min_zoom = target_h / img_h  # Zoom to match height
        
        # Apply padding factor to ensure we can pan/zoom without showing borders
        padding_factor = 1.1  # Add 10% padding
        min_zoom *= padding_factor
        
        # Set zoom range - different for portrait vs landscape for better effect
        if is_portrait:
            zoom_range = random.uniform(0.1, 0.2)  # Less zoom for tall images
        else:
            zoom_range = random.uniform(0.2, 0.3)  # More zoom for wide images
            
        # Randomly decide whether to zoom in or zoom out
        zoom_in = random.choice([True, False])
        
        if zoom_in:
            zoom_start = min_zoom
            zoom_end = zoom_start + zoom_range
        else:
            zoom_end = min_zoom
            zoom_start = zoom_end + zoom_range
        
        # Determine pan direction based on image aspect ratio
        if is_portrait:
            pan_choices = [0, 3, 4]  # Center, top-to-bottom, bottom-to-top (weights vertical)
            weights = [0.2, 0.4, 0.4]  # Higher chance of vertical movement
        elif is_landscape:
            pan_choices = [0, 1, 2]  # Center, left-to-right, right-to-left (weights horizontal)
            weights = [0.2, 0.4, 0.4]  # Higher chance of horizontal movement
        else:  # Square or close to target aspect ratio
            pan_choices = [0, 1, 2, 3, 4]  # All options
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal chance
            
        # Select pan type with weighted random choice
        pan_type = random.choices(pan_choices, weights=weights, k=1)[0]

        # Higher fps for smoother animation
        fps = 60
        
        # Create the clip with its initial frame
        clip = ImageClip(img_array).with_duration(duration)
        clip.fps = fps
        
        # Define a smooth frame generator function
        def get_frame(t):
            # Normalize time to 0-1 range
            progress = t / duration if duration > 0 else 0
            
            # Improved easing function for smoother motion
            # Cubic easing function (smoother than sine)
            progress = progress * progress * (3 - 2 * progress)
            
            # Calculate current zoom level with high precision
            current_zoom = zoom_start + (zoom_end - zoom_start) * progress
            
            # Calculate new dimensions with zoom (maintain float precision)
            new_h = img_h * current_zoom
            new_w = img_w * current_zoom
            
            # Convert to int only when needed for resize operation
            new_h_int, new_w_int = int(new_h), int(new_w)
            
            # Resize using high-quality interpolation
            resized = cv2.resize(img_array, (new_w_int, new_h_int), interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate crop position based on pan type and progress (maintain float precision)
            if pan_type == 0:  # Center zoom
                y_center = (new_h - target_h) / 2
                x_center = (new_w - target_w) / 2
                y = y_center
                x = x_center
            elif pan_type == 1:  # Left to right
                y = (new_h - target_h) / 2
                x = (new_w - target_w) * progress
            elif pan_type == 2:  # Right to left
                y = (new_h - target_h) / 2
                x = (new_w - target_w) * (1 - progress)
            elif pan_type == 3:  # Top to bottom
                x = (new_w - target_w) / 2
                y = (new_h - target_h) * progress
            else:  # Bottom to top (pan_type == 4)
                x = (new_w - target_w) / 2
                y = (new_h - target_h) * (1 - progress)
            
            # Convert to int for cropping, ensuring we stay within bounds
            x_int = max(0, min(int(x), new_w_int - target_w))
            y_int = max(0, min(int(y), new_h_int - target_h))
            
            # Apply crop with precise boundaries
            cropped = resized[y_int:y_int+target_h, x_int:x_int+target_w]
            return cropped
            
        clip.get_frame = get_frame
        return clip
        
    except Exception as e:
        logger.error(f"Error creating Ken Burns clip for {image_path}: {e}")
        # Fallback handling for more resilience
        try:
            # Load image with OpenCV 
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preserve aspect ratio while ensuring the image fills the frame
            h, w = img.shape[:2]
            img_aspect = w / h
            target_aspect = target_size[0] / target_size[1]
            
            if img_aspect > target_aspect:  # Image is wider than target
                # Scale to match height
                new_h = target_size[1]
                new_w = int(new_h * img_aspect)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Crop center to match target width
                x_offset = (new_w - target_size[0]) // 2
                img = img[:, x_offset:x_offset+target_size[0]]
                
            else:  # Image is taller than target
                # Scale to match width
                new_w = target_size[0]
                new_h = int(new_w / img_aspect)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Crop center to match target height
                y_offset = (new_h - target_size[1]) // 2
                img = img[y_offset:y_offset+target_size[1], :]
                
            # Create static clip with the properly sized image
            return ImageClip(img, duration=duration)
            
        except Exception as inner_e:
            logger.error(f"Failed to create fallback static clip for {image_path}: {inner_e}")
            # Last resort: create a blank clip
            blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            return ImageClip(blank, duration=duration)


def process_video_task(request_data: VideoData):
    """
    Downloads assets, generates video using moviepy, uploads result.
    """
    logger.info("Starting video processing task...")
    output_video_filename = f"output_{os.urandom(8).hex()}.mp4"
    final_video_url = None # Initialize

    # Create local output directory if it doesn't exist
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    local_output_path = os.path.join(LOCAL_OUTPUT_DIR, output_video_filename)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_video_path = os.path.join(tmpdir, output_video_filename)
        temp_audio_path = os.path.join(tmpdir, 'temp-audio.m4a')
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
                    bg_music = bg_music.subclipped(0, voice_audio.duration)
                elif bg_music.duration < voice_audio.duration:
                    # Loop background music if shorter than video
                    bg_music = bg_music.loop(duration=voice_audio.duration)
                
                # Store final duration
                video_duration = voice_audio.duration
                logger.info(f"Voice audio duration: {video_duration:.2f} seconds")
                
                # Apply volume adjustment directly to the audio array
                bg_music = bg_music.with_effects([MultiplyVolume(BG_MUSIC_VOLUME)])
            except Exception as e:
                logging.error(f"Error loading audio file: {e}")
                raise

            # --- Create Image Sequence Video ---
            logger.info("Creating image sequence with Ken Burns effect...")
            clips = []
            current_time = 0
            image_index = 0
            
            # Create clips for all images except the last one with fixed duration
            while current_time < video_duration and image_index < len(image_paths) - 1:
                img_path = image_paths[image_index]
                clip_duration = min(IMAGE_DURATION_S, video_duration - current_time)
                if clip_duration <= 0: break # Avoid zero duration clips

                try:
                    kb_clip = create_ken_burns_clip(img_path, clip_duration, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    kb_clip = kb_clip.with_start(current_time) # Not strictly needed for concatenation but good practice
                    clips.append(kb_clip)
                    logger.debug(f"Created clip for image {image_index}: start={current_time:.2f}, duration={clip_duration:.2f}")
                except Exception as e:
                    logger.error(f"Error creating Ken Burns clip for {img_path}: {e}. Skipping image.")
                    # Add a blank clip or handle differently?
                    # For now, just skip, might shorten video.

                current_time += clip_duration
                image_index += 1
            
            # Handle the last image - extend it to fill remaining duration
            if image_index < len(image_paths) and current_time < video_duration:
                last_img_path = image_paths[-1]
                remaining_duration = video_duration - current_time
                
                try:
                    last_clip = create_ken_burns_clip(last_img_path, remaining_duration, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    last_clip = last_clip.with_start(current_time)
                    clips.append(last_clip)
                    logger.debug(f"Created clip for last image: start={current_time:.2f}, duration={remaining_duration:.2f}")
                except Exception as e:
                    logger.error(f"Error creating Ken Burns clip for last image {last_img_path}: {e}")
                    # Try to add a static image as fallback
                    try:
                        img = cv2.imread(last_img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                        static_clip = ImageClip(img, duration=remaining_duration).with_start(current_time)
                        clips.append(static_clip)
                    except Exception as inner_e:
                        logger.error(f"Failed to create static clip for last image: {inner_e}")

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
                    # Make sure to use an integer for font_size and stroke_width
                    txt_clip = TextClip(
                        TEXT_FONT,  # First parameter is font
                        text=item.words,  # Use 'text' parameter
                        font_size=int(TEXT_FONT_SIZE),  # Convert to integer
                        color=TEXT_COLOR,
                        stroke_color=TEXT_STROKE_COLOR,
                        stroke_width=int(TEXT_STROKE_WIDTH),  # Convert to integer
                        size=(int(VIDEO_WIDTH * 0.9), None),  # Convert width to integer
                        method='caption'  # Auto-wrap text
                    )
                    
                    # Calculate position
                    pos_x, pos_y = TEXT_POSITION
                    clip_w, clip_h = txt_clip.size
                    if pos_y == 'bottom':
                         final_pos_y = int(VIDEO_HEIGHT - clip_h - TEXT_MARGIN_BOTTOM)  # Convert to integer
                    elif pos_y == 'center':
                         final_pos_y = int((VIDEO_HEIGHT - clip_h) / 2)  # Convert to integer
                    else: # Assume top or numeric
                         final_pos_y = int(pos_y) if isinstance(pos_y, (int, float)) else pos_y

                    if pos_x == 'center':
                         final_pos_x = int((VIDEO_WIDTH - clip_w) / 2)  # Convert to integer
                    else: # Assume left/right or numeric
                        final_pos_x = int(pos_x) if isinstance(pos_x, (int, float)) else pos_x

                    txt_clip = txt_clip.with_position((final_pos_x, final_pos_y))
                    txt_clip = txt_clip.with_start(start_time)
                    txt_clip = txt_clip.with_duration(text_duration)
                    text_clips.append(txt_clip)
                    logger.debug(f"Created text clip: '{item.words}' start={start_time:.2f} end={end_time:.2f}")
                except Exception as e:
                    logger.error(f"Error creating text clip for '{item.words}': {e}")
                    # Continue without this text clip rather than failing the entire process
                    continue

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
                temp_audiofile=temp_audio_path, # Ensure temp file is in tmpdir
                remove_temp=True,
                fps=24,
                bitrate="5000k",  # Higher bitrate for better quality
                preset='medium', # 'medium' is a balance, 'fast' or 'ultrafast' for speed
                threads=4,
                logger='bar' # or None to disable progress bar logging
            )
            logger.info("Video file written successfully.")
            
            # Copy files to local output directory for development
            logger.info(f"Copying output video to local directory: {local_output_path}")
            shutil.copy2(output_video_path, local_output_path)
            
            # Copy the audio file if it exists
            if os.path.exists(voice_path):
                voice_audio_filename = os.path.basename(voice_path)
                local_voice_path = os.path.join(LOCAL_OUTPUT_DIR, voice_audio_filename)
                shutil.copy2(voice_path, local_voice_path)
                logger.info(f"Copied voice audio to: {local_voice_path}")
            
            if os.path.exists(bg_music_path):
                bg_music_filename = os.path.basename(bg_music_path)
                local_bg_music_path = os.path.join(LOCAL_OUTPUT_DIR, bg_music_filename)
                shutil.copy2(bg_music_path, local_bg_music_path)
                logger.info(f"Copied background music to: {local_bg_music_path}")

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

@app.post("/test-animation")
async def test_animation(request: AnimationTestRequest, image_count: int = Query(1, ge=1)):
    """
    Test endpoint to process images with various animation effects without audio.
    
    Args:
        request: Contains image URLs, animation effect type, and parameters
        image_count: Number of images to process (defaults to 1)
    
    Returns:
        JSON with the path to the generated video file
    """
    logger.info(f"Received animation test request for {image_count} images with effect: {request.effect_type}")
    
    # Limit the number of images to process
    image_urls = request.image_urls[:image_count]
    if not image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")
    
    # Generate a unique filename
    output_video_filename = f"{request.effect_type}_test_{os.urandom(8).hex()}.mp4"
    
    # Create local output directory if it doesn't exist
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    local_output_path = os.path.join(LOCAL_OUTPUT_DIR, output_video_filename)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Download images
            logger.info("Downloading images...")
            image_paths = [download_file(str(url), tmpdir) for url in image_urls]
            
            # Create clips with the requested animation effect
            logger.info(f"Creating clips with {request.effect_type} effect...")
            clips = []
            duration_per_image = request.duration_per_image
            effect_params = request.effect_params or {}
            
            for i, img_path in enumerate(image_paths):
                try:
                    # Apply the appropriate animation effect based on effect_type
                    if request.effect_type == "ken_burns":
                        animated_clip = create_ken_burns_clip(img_path, duration_per_image, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    # Add more effect types here as they become available
                    # elif request.effect_type == "zoom":
                    #     animated_clip = create_zoom_clip(img_path, duration_per_image, (VIDEO_WIDTH, VIDEO_HEIGHT), **effect_params)
                    # elif request.effect_type == "pan":
                    #     animated_clip = create_pan_clip(img_path, duration_per_image, (VIDEO_WIDTH, VIDEO_HEIGHT), **effect_params)
                    else:
                        raise ValueError(f"Unsupported animation effect: {request.effect_type}")
                        
                    animated_clip = animated_clip.with_start(i * duration_per_image)
                    clips.append(animated_clip)
                    logger.info(f"Created {request.effect_type} clip for image {i+1}")
                except Exception as e:
                    logger.error(f"Error creating animation clip for {img_path}: {e}")
                    # Try a simpler static clip as fallback
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                        static_clip = ImageClip(img, duration=duration_per_image).with_start(i * duration_per_image)
                        clips.append(static_clip)
                        logger.info(f"Created static clip for image {i+1} as fallback")
                    except Exception as inner_e:
                        logger.error(f"Failed to create fallback static clip: {inner_e}")
            
            if not clips:
                raise HTTPException(status_code=500, detail="Failed to create any video clips from images")
            
            # Concatenate clips
            logger.info("Concatenating clips...")
            total_duration = len(clips) * duration_per_image
            final_video = concatenate_videoclips(clips).with_duration(total_duration)
            
            # Write video to file
            logger.info(f"Writing video to {local_output_path}")
            temp_audio_path = os.path.join(tmpdir, 'temp-audio.m4a')
            final_video.write_videofile(
                local_output_path,
                codec='libx264',
                audio=False,  # No audio for this test
                temp_audiofile=temp_audio_path,
                remove_temp=True,
                fps=24,
                bitrate="5000k",  # Higher bitrate for better quality
                preset='medium',
                threads=4,
                logger='bar'
            )
            
            # Close clips
            for clip in clips:
                if clip: clip.close()
            if final_video: final_video.close()
            
            logger.info(f"Animation test video created at {local_output_path}")
            return {
                "message": f"{request.effect_type} animation test video created successfully", 
                "video_path": local_output_path,
                "effect_type": request.effect_type,
                "duration": total_duration
            }
            
        except Exception as e:
            logger.exception(f"Error in animation test: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing animation test: {str(e)}")

# --- Uvicorn Runner (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server locally on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

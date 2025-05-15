import os
import json
import tempfile
import logging
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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
from datetime import datetime

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
TEXT_FONT = 'fonts/Lexend/static/Lexend-Bold.ttf' # Full path to the Lexend Bold font file
TEXT_FONT_SIZE = 70 # Increased font size for better visibility
TEXT_COLOR = 'white'
TEXT_STROKE_COLOR = 'black'
TEXT_STROKE_WIDTH = 5 # Increased for more visible stroke

TEXT_POSITION = ('center', 'center') # Updated to center the text in the transparent area
TEXT_MARGIN_BOTTOM = 120 # Increased to avoid watermark overlap

# Shadow effect parameters
TEXT_SHADOW_COLOR = 'black'
TEXT_SHADOW_OFFSET = (1, 1) # Shadow offset in pixels (x, y)
TEXT_SHADOW_TYPE = 'radial' # Default shadow type: 'drop' or 'radial'
TEXT_RADIAL_SHADOW_BLUR = 20 # Blur amount for radial shadow (higher values = more blur)

TEXT_SHADOW_OPACITY = 0.8 # Opacity of the shadow (0.0 to 1.0, higher = darker)

# Shadow layer count for thickness
TEXT_SHADOW_LAYERS_DROP = 1 # Number of shadow layers for drop shadow
TEXT_SHADOW_LAYERS_RADIAL = 5 # Number of shadow layers for radial shadow

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
    shadow_type: Optional[str] = None # Optional shadow type ('drop' or 'radial')

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

def create_drop_shadow(txt_clip, text, final_pos_x, final_pos_y, start_time, text_duration):
    """
    Creates a drop shadow effect for text using a separate TextClip with offset.
    
    Args:
        txt_clip: The main text clip
        text: The text content
        final_pos_x: X position of the main text
        final_pos_y: Y position of the main text
        start_time: Start time of the clip
        text_duration: Duration of the clip
        
    Returns:
        TextClip: The shadow clip properly positioned and timed
    """
    # Create a separate shadow TextClip with the same parameters but black color
    shadow_clip = TextClip(
        TEXT_FONT,
        text=text,
        font_size=int(TEXT_FONT_SIZE),
        color=TEXT_SHADOW_COLOR,
        stroke_color='black',
        stroke_width=int(TEXT_STROKE_WIDTH) + 2,  # Slightly thicker for shadow
        size=(int(VIDEO_WIDTH * 0.9), None),
        method='caption',
        transparent=True
    )
    
    # Position the shadow with a slight offset
    shadow_offset_x, shadow_offset_y = TEXT_SHADOW_OFFSET
    shadow_pos_x = final_pos_x + shadow_offset_x
    shadow_pos_y = final_pos_y + shadow_offset_y
    
    # Set positions and timing
    shadow_clip = shadow_clip.with_position((shadow_pos_x, shadow_pos_y))
    shadow_clip = shadow_clip.with_start(start_time)
    shadow_clip = shadow_clip.with_duration(text_duration)
    
    return shadow_clip

def create_radial_shadow(txt_clip, text, final_pos_x, final_pos_y, start_time, text_duration):
    """
    Creates a radial shadow effect (glow) for text.
    
    Args:
        txt_clip: The main text clip
        text: The text content
        final_pos_x: X position of the main text
        final_pos_y: Y position of the main text
        start_time: Start time of the clip
        text_duration: Duration of the clip
        
    Returns:
        TextClip: The radial shadow clip properly positioned and timed
    """
    # Calculate padding based on blur radius to ensure shadow has room to render
    # Padding should be at least 2-3 times the blur radius
    padding = max(int(TEXT_RADIAL_SHADOW_BLUR * 3), 100)
    
    # Create a large TextClip with the text in white on transparent background
    # Use white text without stroke - will be used just for the mask
    glow_base = TextClip(
        TEXT_FONT,
        text=text,
        font_size=int(TEXT_FONT_SIZE),
        color='white',  # White text for clear mask
        stroke_color=None,
        stroke_width=0,
        size=(int(VIDEO_WIDTH * 0.9), None),
        method='caption',
        transparent=True
    )
    
    # Get the original size of the text
    orig_w, orig_h = glow_base.size
    
    # Create a larger canvas with padding on all sides
    padded_w = orig_w + padding * 2
    padded_h = orig_h + padding * 2
    
    # Create a blank padded canvas (transparent)
    padded_canvas = np.zeros((padded_h, padded_w, 4), dtype=np.uint8)
    
    # Get the frame as numpy array (this should be RGBA)
    text_frame = glow_base.get_frame(0)
    
    # Place the text in the center of the padded canvas
    if text_frame.shape[2] == 4:  # RGBA
        # Place the text frame in the middle of the padded canvas
        y_offset = padding
        x_offset = padding
        padded_canvas[y_offset:y_offset+orig_h, x_offset:x_offset+orig_w] = text_frame
        
        # Extract the alpha channel for the mask
        mask = padded_canvas[:, :, 3]
    else:  # RGB
        # Place the RGB frame in the middle of the padded canvas
        y_offset = padding
        x_offset = padding
        padded_canvas[y_offset:y_offset+orig_h, x_offset:x_offset+orig_w, :3] = text_frame
        
        # If no alpha channel, use the luminance of the RGB image
        # White text (255,255,255) will have high luminance values
        mask = np.mean(padded_canvas[:, :, :3], axis=2).astype('uint8')
    
    # Convert to PIL Image for processing
    mask_img = Image.fromarray(mask)
    
    # Apply gaussian blur to create the glow effect
    # A higher radius creates a more spread-out glow
    blurred_mask = mask_img.filter(
        ImageFilter.GaussianBlur(radius=TEXT_RADIAL_SHADOW_BLUR)
    )
    
    # Convert blurred mask back to numpy for processing
    blurred_mask_array = np.array(blurred_mask)
    
    # Create RGBA array with shadow color and blurred mask as alpha
    shadow_r, shadow_g, shadow_b = hex_to_rgb(TEXT_SHADOW_COLOR)
    glow_array = np.zeros((padded_h, padded_w, 4), dtype=np.uint8)
    glow_array[:, :, 0] = shadow_r  # R
    glow_array[:, :, 1] = shadow_g  # G
    glow_array[:, :, 2] = shadow_b  # B
    
    # Apply the opacity factor to the alpha channel
    # Lower opacity (closer to 0) makes the shadow darker and more intense
    # Higher opacity (closer to 1) makes the shadow lighter and more transparent
    opacity_factor = max(0.1, min(1.0, TEXT_SHADOW_OPACITY))  # Clamp between 0.1 and 1.0
    glow_array[:, :, 3] = (blurred_mask_array * opacity_factor).astype(np.uint8)
    
    # Create the final glow clip
    glow_clip = ImageClip(glow_array).with_duration(text_duration)
    
    # Adjust position to account for the padding
    # The text will remain in the same position, but the shadow extends beyond it
    shadow_pos_x = final_pos_x - padding
    shadow_pos_y = final_pos_y - padding
    
    glow_clip = glow_clip.with_position((shadow_pos_x, shadow_pos_y))
    glow_clip = glow_clip.with_start(start_time)
    
    return glow_clip

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    # Handle both '#RRGGBB' and 'RRGGBB' formats
    hex_color = hex_color.lstrip('#')
    
    # Handle named colors
    if isinstance(hex_color, str) and not hex_color.startswith('#') and not all(c in '0123456789ABCDEFabcdef' for c in hex_color):
        # For named colors like 'black', return appropriate RGB
        color_map = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'transparent': (0, 0, 0, 0)
        }
        return color_map.get(hex_color.lower(), (0, 0, 0))
    
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_text_clip(text, start_time, end_time, shadow_type=None):
    """
    Creates a text clip with the specified shadow effect.
    
    Args:
        text (str): The text to display
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        shadow_type (str, optional): Type of shadow effect: 'drop' or 'radial'.
            If None, uses the default TEXT_SHADOW_TYPE
        
    Returns:
        list: List of clips (shadow and main text)
    """
    # Use default shadow type if not specified
    if shadow_type is None:
        shadow_type = TEXT_SHADOW_TYPE
        
    text_duration = end_time - start_time
    if text_duration <= 0:
        logger.warning(f"Skipping text clip with zero/negative duration: '{text}'")
        return []

    clips = []
    try:
        # Create the main text clip with transparent background
        txt_clip = TextClip(
            TEXT_FONT,  # Font path
            text=text,
            font_size=int(TEXT_FONT_SIZE),
            color=TEXT_COLOR,
            stroke_color=TEXT_STROKE_COLOR,
            stroke_width=int(TEXT_STROKE_WIDTH),
            size=(int(VIDEO_WIDTH * 0.9), None),
            method='caption',  # Auto-wrap text
            transparent=True  # Ensure transparency
        )
        
        # Get the size of the text clip
        clip_w, clip_h = txt_clip.size
        
        # Calculate position
        pos_x, pos_y = TEXT_POSITION
        if pos_y == 'bottom':
            final_pos_y = int(VIDEO_HEIGHT - clip_h - TEXT_MARGIN_BOTTOM)
        elif pos_y == 'center':
            final_pos_y = int((VIDEO_HEIGHT - clip_h) / 2)
        else:  # Assume top or numeric
            final_pos_y = int(pos_y) if isinstance(pos_y, (int, float)) else pos_y

        if pos_x == 'center':
            final_pos_x = int((VIDEO_WIDTH - clip_w) / 2)
        else:  # Assume left/right or numeric
            final_pos_x = int(pos_x) if isinstance(pos_x, (int, float)) else pos_x
        
        # Create shadow effect based on selected type
        if shadow_type == 'drop':
            shadow_clip = create_drop_shadow(txt_clip, text, final_pos_x, final_pos_y, start_time, text_duration)
            
            # Add the drop shadow layer(s)
            shadow_layers = TEXT_SHADOW_LAYERS_DROP
            for _ in range(shadow_layers):
                clips.append(shadow_clip)
                
        elif shadow_type == 'radial':
            shadow_clip = create_radial_shadow(txt_clip, text, final_pos_x, final_pos_y, start_time, text_duration)
            
            # Add the radial shadow layer(s)
            shadow_layers = TEXT_SHADOW_LAYERS_RADIAL
            for _ in range(shadow_layers):
                clips.append(shadow_clip)
                
        else:
            logger.warning(f"Unknown shadow type '{shadow_type}', falling back to drop shadow")
            shadow_clip = create_drop_shadow(txt_clip, text, final_pos_x, final_pos_y, start_time, text_duration)
            
            # Add the default shadow layer
            clips.append(shadow_clip)
        
        # Set position and timing for the main text clip
        txt_clip = txt_clip.with_position((final_pos_x, final_pos_y))
        txt_clip = txt_clip.with_start(start_time)
        txt_clip = txt_clip.with_duration(text_duration)
        
        # Add the text clip on top
        clips.append(txt_clip)
        
        shadow_type_str = shadow_type if shadow_type in ('drop', 'radial') else 'unknown'
        logger.debug(f"Created text clip with {shadow_type_str} shadow: '{text}' start={start_time:.2f} end={end_time:.2f}")
    except Exception as e:
        logger.error(f"Error creating text clip for '{text}': {e}")
        # Continue without this text clip rather than failing the entire process
    
    return clips

def process_audio(voice_path, bg_music_path, target_duration=None):
    """
    Process audio files by loading, adjusting, and combining them.
    
    Args:
        voice_path (str): Path to voice/narration audio file
        bg_music_path (str): Path to background music file
        target_duration (float, optional): Target duration to limit audio
        
    Returns:
        tuple: (final_composite_audio, actual_duration)
    """
    try:
        # Load voice audio
        voice_audio = AudioFileClip(voice_path)
        
        # Subclip if target duration specified
        if target_duration is not None:
            voice_audio = voice_audio.subclipped(0, target_duration)
            
        # Get actual duration (might be less than target if audio is shorter)
        actual_duration = voice_audio.duration
        logger.info(f"Voice audio duration: {actual_duration:.2f} seconds")
        
        # Load background music
        bg_music = AudioFileClip(bg_music_path)
        
        # Adjust background music duration
        if bg_music.duration > actual_duration:
            bg_music = bg_music.subclipped(0, actual_duration)
        elif bg_music.duration < actual_duration:
            # Loop background music if shorter than video
            bg_music = bg_music.loop(duration=actual_duration)
        
        # Apply volume adjustment to background music
        bg_music = bg_music.with_effects([MultiplyVolume(BG_MUSIC_VOLUME)])
        
        # Combine audio tracks
        final_audio = CompositeAudioClip([voice_audio, bg_music])
        
        # Ensure final audio has the exact duration
        final_audio = final_audio.with_duration(actual_duration)
        
        return final_audio, actual_duration
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise

def process_video_task(request_data: VideoData, shadow_type: Optional[str] = None):
    """
    Downloads assets, generates video using moviepy, uploads result.
    """
    logger.info("Starting video processing task...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"output_{timestamp}.mp4"
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

            # 2. Process Audio
            logging.info("Processing audio...")
            try:
                final_audio, video_duration = process_audio(voice_path, bg_music_path)
            except Exception as e:
                logging.error(f"Error processing audio: {e}")
                raise

            # 3. Create Image Sequence Video
            logger.info("Creating image sequence...")
            final_image_video = create_image_sequence(image_paths, video_duration, IMAGE_DURATION_S)

            # 4. Create Text Overlays
            logger.info("Creating text overlays...")
            text_clips = create_text_overlays_from_transcripts(transcripts_list, max_duration=video_duration, shadow_type=shadow_type)

            # 5. Composite Video
            logger.info("Compositing final video...")
            video_with_text = CompositeVideoClip([final_image_video] + text_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            video_with_text = video_with_text.with_duration(video_duration)
            final_clip = video_with_text.with_audio(final_audio)

            # 6. Render Video
            render_video(final_clip, output_video_path, temp_audio_path, quality='medium')
            
            # 7. Copy files to local output directory for development
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

            # 8. Upload Result
            logger.info(f"Uploading result: {output_video_path}")
            final_video_url = upload_to_gcs(output_video_path, OUTPUT_BUCKET_NAME, output_video_filename)
            logger.info(f"Video processing complete. Final URL: {final_video_url}")

        except (HTTPException, ValueError, requests.exceptions.RequestException) as e:
            logger.error(f"Error during video processing task: {e}")
        except Exception as e:
            logger.exception(f"Unhandled exception during video processing task: {e}")
        finally:
            # Clean up all clips
            cleanup_clips(
                final_audio if 'final_audio' in locals() else None,
                final_image_video if 'final_image_video' in locals() else None,
                video_with_text if 'video_with_text' in locals() else None,
                final_clip if 'final_clip' in locals() else None,
                text_clips if 'text_clips' in locals() else None
            )
                
            logger.info(f"Finished processing. Cleaned up temporary directory: {tmpdir}")
            if final_video_url:
                logger.info(f"Task succeeded. Video available at: {final_video_url}")
            else:
                logger.error("Task failed to produce a video URL.")

def render_video(final_clip, output_path, temp_audio_path, quality='medium'):
    """
    Renders a video clip to a file with appropriate settings.
    
    Args:
        final_clip: The MoviePy clip to render
        output_path (str): Where to save the output video
        temp_audio_path (str): Path for temporary audio file
        quality (str): Rendering quality preset ('ultrafast', 'medium', 'high')
        
    Returns:
        str: Path to the rendered video file
    """
    try:
        # Set quality-dependent parameters
        quality_presets = {
            'ultrafast': {
                'preset': 'ultrafast',
                'bitrate': '3000k',
                'threads': 4
            },
            'medium': {
                'preset': 'medium',
                'bitrate': '5000k',
                'threads': 4
            },
            'high': {
                'preset': 'slow',
                'bitrate': '8000k',
                'threads': 4
            }
        }
        
        preset = quality_presets.get(quality, quality_presets['medium'])
        
        logger.info(f"Rendering video to {output_path} with {quality} quality...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            fps=24,
            bitrate=preset['bitrate'],
            preset=preset['preset'],
            threads=preset['threads'],
            logger='bar'
        )
        logger.info(f"Video successfully rendered to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error rendering video: {e}")
        raise

def create_text_overlays_from_transcripts(transcripts, max_duration=None, shadow_type=None):
    """
    Creates text overlays from transcript items.
    
    Args:
        transcripts (list): List of TranscriptItem objects
        max_duration (float, optional): Maximum duration to include
        shadow_type (str, optional): Type of shadow to use ('drop' or 'radial').
            If None, uses the default TEXT_SHADOW_TYPE
        
    Returns:
        list: List of text clips
    """
    text_clips = []
    
    # Filter transcripts if max_duration specified
    filtered_transcripts = transcripts
    if max_duration is not None:
        filtered_transcripts = [
            item for item in transcripts 
            if item.start < max_duration
        ]
        
        # For transcript items that extend beyond max_duration, cap their end time
        for item in filtered_transcripts:
            if item.end > max_duration:
                item.end = max_duration
    
    logger.info(f"Creating text overlays for {len(filtered_transcripts)} transcript items with {shadow_type or TEXT_SHADOW_TYPE} shadow")
    
    # Create text clips for each transcript item
    for item in filtered_transcripts:
        clips = create_text_clip(item.words, item.start, item.end, shadow_type)
        text_clips.extend(clips)
    
    return text_clips

def cleanup_clips(*clips_lists):
    """
    Safely close all clips to prevent memory leaks.
    
    Args:
        *clips_lists: Variable number of clip lists or individual clips
    """
    for clips_or_clip in clips_lists:
        if not clips_or_clip:
            continue
            
        if isinstance(clips_or_clip, list):
            for clip in clips_or_clip:
                if clip:
                    try:
                        clip.close()
                    except Exception as e:
                        logger.warning(f"Error closing clip: {e}")
        else:
            try:
                clips_or_clip.close()
            except Exception as e:
                logger.warning(f"Error closing clip: {e}")

def create_image_sequence(image_paths, total_duration, image_duration=IMAGE_DURATION_S):
    """
    Creates a sequence of image clips with Ken Burns effect.
    
    Args:
        image_paths (list): List of image file paths
        total_duration (float): Total video duration in seconds
        image_duration (float, optional): Duration per image
        
    Returns:
        moviepy.editor.VideoClip: Concatenated video clip of the image sequence
    """
    if not image_paths:
        raise ValueError("No image paths provided")
        
    logger.info(f"Creating image sequence with {len(image_paths)} images for total duration of {total_duration:.2f}s")
    
    clips = []
    current_time = 0
    image_index = 0
    
    # Create clips for all images except the last one with fixed duration
    while current_time < total_duration and image_index < len(image_paths) - 1:
        img_path = image_paths[image_index]
        clip_duration = min(image_duration, total_duration - current_time)
        if clip_duration <= 0: 
            break  # Avoid zero duration clips

        try:
            kb_clip = create_ken_burns_clip(img_path, clip_duration, (VIDEO_WIDTH, VIDEO_HEIGHT))
            kb_clip = kb_clip.with_start(current_time)
            clips.append(kb_clip)
            logger.debug(f"Created clip for image {image_index}: start={current_time:.2f}, duration={clip_duration:.2f}")
        except Exception as e:
            logger.error(f"Error creating Ken Burns clip for {img_path}: {e}. Skipping image.")

        current_time += clip_duration
        image_index += 1
    
    # Handle the last image - extend it to fill remaining duration
    if image_index < len(image_paths) and current_time < total_duration:
        last_img_path = image_paths[-1]
        remaining_duration = total_duration - current_time
        
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
                logger.error(f"Failed to create fallback static clip for last image: {inner_e}")

    if not clips:
        raise ValueError("Failed to create any video clips from images.")

    final_video = concatenate_videoclips(clips).with_duration(total_duration)
    logger.info(f"Created image sequence video with duration: {final_video.duration:.2f} seconds")
    
    return final_video

# --- API Endpoint ---
@app.post("/")
async def create_short_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    API endpoint to trigger YouTube Shorts video generation.
    Accepts POST requests with asset URLs and transcript data.
    Initiates a background task for processing.
    
    Optional Parameters:
        shadow_type: Type of shadow to use for text ('drop' or 'radial')
    """
    logger.info(f"Received request for type: {request.type} with shadow_type: {request.shadow_type or TEXT_SHADOW_TYPE}")
    if request.type == "ImageVideo":
        # Add the processing to background tasks
        background_tasks.add_task(process_video_task, request.data, request.shadow_type)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"{request.effect_type}_test_{timestamp}.mp4"
    
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
            temp_audio_path = os.path.join(tmpdir, 'temp-audio.m4a')
            render_video(final_video, local_output_path, temp_audio_path, quality='medium')
            
            # Clean up resources
            cleanup_clips(clips, final_video)
            
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

@app.post("/test-chunk")
async def test_chunk_video(request: VideoRequest, shadow_type: str = Query(None, description="Shadow type to use: 'drop' or 'radial'")):
    """
    API endpoint for rapid prototyping that generates a short 6-second video with:
    - Just the first image with Ken Burns effect
    - First 6 seconds of voice/audio
    - Captions that match that timeframe
    - Background music
    - Optional shadow type for text
    
    Query Parameters:
        shadow_type: Optional shadow type ('drop' or 'radial'). Uses default if not specified.
    
    Returns a path to the generated video for preview purposes.
    """
    logger.info(f"Received test chunk request for type: {request.type} with shadow_type: {shadow_type or TEXT_SHADOW_TYPE}")
    if request.type != "ImageVideo":
        raise HTTPException(status_code=400, detail=f"Unsupported type: {request.type}. Only 'ImageVideo' is supported.")
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"test_chunk_{timestamp}.mp4"
    
    # Create local output directory if it doesn't exist
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    local_output_path = os.path.join(LOCAL_OUTPUT_DIR, output_video_filename)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_video_path = os.path.join(tmpdir, output_video_filename)
        temp_audio_path = os.path.join(tmpdir, 'temp-audio.m4a')
        try:
            logger.info(f"Using temporary directory: {tmpdir}")
            
            # Download only the required assets
            logger.info("Downloading assets for test chunk...")
            voice_path = download_file(str(request.data.voice_url), tmpdir)
            bg_music_path = download_file(str(request.data.background_music_url), tmpdir)
            
            # Download only the first image
            first_image_path = None
            if request.data.image_urls:
                first_image_path = download_file(str(request.data.image_urls[0]), tmpdir)
            else:
                raise HTTPException(status_code=400, detail="No image URLs provided")
            
            # Limit test chunk to x seconds
            TEST_DURATION = 2.0
            
            # Process audio with limited duration
            final_audio, test_duration = process_audio(voice_path, bg_music_path, TEST_DURATION)
            
            # Create Ken Burns clip for the first image
            logger.info("Creating Ken Burns effect for test image...")
            kb_clip = create_ken_burns_clip(first_image_path, test_duration, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            # Create text overlays for the filtered transcripts
            logger.info(f"Creating text overlays with {shadow_type or TEXT_SHADOW_TYPE} shadow...")
            text_clips = create_text_overlays_from_transcripts(request.data.transcripts, test_duration, shadow_type)
            
            # Composite video
            logger.info("Compositing final test video...")
            video_with_text = CompositeVideoClip([kb_clip] + text_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            video_with_text = video_with_text.with_duration(test_duration)
            final_clip = video_with_text.with_audio(final_audio)
            
            # Render with faster settings for quick preview
            render_video(final_clip, output_video_path, temp_audio_path, quality='ultrafast')
            
            # Copy to output directory
            logger.info(f"Copying test chunk to {local_output_path}")
            shutil.copy2(output_video_path, local_output_path)
            
            # Clean up clips
            cleanup_clips(final_audio, kb_clip, video_with_text, final_clip, text_clips)
            
            return {
                "message": "Test chunk video created successfully",
                "video_path": local_output_path,
                "duration": test_duration,
                "shadow_type": shadow_type or TEXT_SHADOW_TYPE
            }
            
        except Exception as e:
            logger.exception(f"Error in test chunk generation: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating test chunk: {str(e)}")

# --- Uvicorn Runner (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server locally on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

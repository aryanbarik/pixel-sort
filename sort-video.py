import cv2
import numpy as np
import os
import tempfile
import shutil
from moviepy.editor import VideoFileClip, ImageSequenceClip

def frame_pixel_sort(frame, canny_low=50, canny_high=150, 
                    sort_key='brightness', sort_method='row',
                    region_sort_direction='horizontal',
                    min_region_area=100):
    """Pixel sort a single video frame (BGR format)"""
    # Convert BGR to HSV for sorting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Edge detection (Canny)
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), canny_low, canny_high)
    
    # Choose sorting criterion
    if sort_key == 'hue':
        sort_channel = h
    elif sort_key == 'saturation':
        sort_channel = s
    else:  # brightness
        sort_channel = v

    # Apply selected sorting method
    if sort_method == 'row':
        v, h, s = _sort_rows(v, h, s, edges, sort_channel)
    elif sort_method == 'column':
        v, h, s = _sort_columns(v, h, s, edges, sort_channel)
    elif sort_method == 'gradient':
        v, h, s = _sort_gradient(frame, v, h, s, edges, sort_channel)
    elif sort_method == 'region':
        v, h, s = _sort_regions(v, h, s, edges, sort_channel,
                               sort_direction=region_sort_direction,
                               min_region_area=min_region_area)
    
    # Merge channels and convert back to BGR
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def process_video(input_path, output_path, 
                canny_low=50, canny_high=150,
                sort_key='brightness', sort_method='row',
                region_sort_direction='horizontal',
                min_region_area=100):
    """
    Process a video with pixel sorting while preserving audio
    
    :param input_path: Path to input video file
    :param output_path: Path to save processed video
    :param sort_method: Sorting method (row/column/gradient/region)
    :param region_sort_direction: For region method: horizontal/vertical
    :param min_region_area: Minimum area for regions to be processed
    """
    # Extract original audio
    video_clip = VideoFileClip(input_path)
    audio = video_clip.audio
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read video with OpenCV
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process and save frames
        processed_frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = frame_pixel_sort(
                frame,
                canny_low=canny_low,
                canny_high=canny_high,
                sort_key=sort_key,
                sort_method=sort_method,
                region_sort_direction=region_sort_direction,
                min_region_area=min_region_area
            )
            
            # Convert BGR to RGB for MoviePy
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_frame_rgb)

            # Print progress
            print(f"Processed frame {i+1}/{frame_count}")

        cap.release()

        # Create video clip from processed frames
        processed_clip = ImageSequenceClip(processed_frames, fps=fps)
        
        # Add original audio
        processed_clip = processed_clip.set_audio(audio)
        
        # Write final video
        processed_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            logger=None  # Disable progress messages for cleaner output
        )

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

# Include the _sort_rows, _sort_columns, _sort_gradient, and _sort_regions 
# functions from previous implementation here

if __name__ == "__main__":
    # Example usage
    process_video(
        input_path="input_video.mp4",
        output_path="sorted_video.mp4",
        canny_low=30,
        canny_high=100,
        sort_key='hue',
        sort_method='region',
        region_sort_direction='vertical',
        min_region_area=500
    )
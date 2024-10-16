import os
import json
import cv2
import hashlib
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import UploadFile
import aiofiles

# Path configurations
VIDEO_DIRECTORY = "generated_videos"
UPLOAD_DIRECTORY = "uploaded_files"
MAPPING_DIRECTORY = "mappings"
MAPPING_FILE = f"{MAPPING_DIRECTORY}/color_mapping.json"
METADATA_FILE = f"{MAPPING_DIRECTORY}/video_metadata.json"

# Ensure directories exist
os.makedirs(VIDEO_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(MAPPING_DIRECTORY, exist_ok=True)

NUM_THREADS = 8  # Number of threads

# Load the color mapping from JSON
def load_color_mapping():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save the color mapping to JSON
def save_color_mapping(mapping):
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f)

# Save metadata to a JSON file
def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

# Function to convert an uploaded file into a video
async def convert_file_to_video(file: UploadFile):
    try:
        # Save uploaded file asynchronously
        temp_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            await temp_file.write(content)

        # Calculate file hash (SHA-256)
        file_hash = hashlib.sha256(content).hexdigest()
        total_bytes = len(content)

        # Video path where the encoded video will be saved
        video_path = os.path.join(VIDEO_DIRECTORY, "sample_video.mp4")

        # Video writer configuration (8K, 30 FPS)
        frame_height = 4320
        frame_width = 7680
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

        # Black buffer for the first second (30 frames)
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for _ in range(30):
            out.write(black_frame)

        # Color mapping
        color_mapping = {}
        byte_to_block_map = {}  # Track which byte corresponds to which frame/block

        for i in range(total_bytes):
            r = (i * 3) % 256
            g = (i * 5) % 256
            b = (i * 7) % 256
            color_mapping[i] = {"r": r, "g": g, "b": b}
            byte_to_block_map[i] = {"frame": 6, "block": i}  # Assign block information

        # Use threading to generate video frames in parallel
        def process_frame(frame_idx):
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            if frame_idx == 5:  # Use frame 6 for color encoding
                known_byte_index = total_bytes // 2
                r = color_mapping[known_byte_index]["r"]
                g = color_mapping[known_byte_index]["g"]
                b = color_mapping[known_byte_index]["b"]

                # Set a 4x4 block in the middle with the selected color
                x_start = frame_width // 2 - 2
                y_start = frame_height // 2 - 2
                frame[y_start:y_start + 4, x_start:x_start + 4] = (r, g, b)
            return frame

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(process_frame, i) for i in range(30)]
            for future in as_completed(futures):
                frame = future.result()
                out.write(frame)

        out.release()

        # Save color mapping and metadata
        save_color_mapping(color_mapping)
        metadata = {
            "file_size": total_bytes,
            "file_hash": file_hash,
            "original_format": file.filename.split('.')[-1],
            "video_path": video_path,
            "byte_ranges": byte_to_block_map  # Include byte to block mapping in metadata
        }
        save_metadata(metadata)

        # Load and log metadata to verify it
        with open(METADATA_FILE, 'r') as f:
            loaded_metadata = json.load(f)
            logger.info(f"Loaded Metadata: {loaded_metadata}")
        
            # Check the structure of byte_ranges
            if 'byte_ranges' in loaded_metadata:
                for byte_index in range(total_bytes):
                    if str(byte_index) not in loaded_metadata['byte_ranges']:
                        logger.warning(f"Missing byte mapping for byte index: {byte_index}")
                    else:
                        logger.debug(f"Byte {byte_index} mapped to frame {loaded_metadata['byte_ranges'][str(byte_index)]['frame']} and block {loaded_metadata['byte_ranges'][str(byte_index)]['block']}")
            else:
                logger.warning("No byte_ranges found in metadata.")

        logger.info(f"Video created successfully: {video_path}")
        return video_path

    except Exception as e:
        logger.error(f"Error converting file to video: {str(e)}")
        return None


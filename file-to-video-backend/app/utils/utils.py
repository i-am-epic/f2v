import os
import json
import cv2
from loguru import logger
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Loguru for detailed logging
logger.add(sys.stderr, level="DEBUG", format="{time} - {level} - {message}")

# Path configurations
RECONSTRUCTED_FILE_PATH = "reconstructed_file.bin"
VIDEO_DIRECTORY = "generated_videos"
MAPPINGS_DIR = "mappings"
METADATA_PATH = f"{MAPPINGS_DIR}/video_metadata.json"
COLOR_MAPPING_PATH = f"{MAPPINGS_DIR}/color_mapping.json"
NUM_THREADS = 8  # Number of threads for parallel processing

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of the file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hash_sha256.update(byte_block)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    return hash_sha256.hexdigest()

def check_file_properties(metadata):
    """Check file properties against metadata."""
    file_path = RECONSTRUCTED_FILE_PATH
    try:
        actual_file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return

    actual_file_hash = calculate_file_hash(file_path)

    logger.info("Overview of file properties:")
    logger.info(f"Expected File Size: {metadata['file_size']} bytes")
    logger.info(f"Actual File Size: {actual_file_size} bytes")

    if actual_file_size != metadata['file_size']:
        logger.warning("Warning: File size does not match!")
    else:
        logger.info("File size matches.")

    logger.info(f"Expected File Hash: {metadata['file_hash']}")
    logger.info(f"Actual File Hash: {actual_file_hash}")

    if actual_file_hash != metadata['file_hash']:
        logger.warning("Warning: File hash does not match!")
    else:
        logger.info("File hash matches.")

def verify_metadata(metadata):
    """Verify that the metadata contains necessary byte block details."""
    required_keys = ['file_size', 'file_hash', 'byte_ranges']
    
    for key in required_keys:
        if key not in metadata:
            logger.error(f"Missing required metadata key: {key}")
            return False
            
    if not isinstance(metadata['byte_ranges'], dict) or not metadata['byte_ranges']:
        logger.error("Byte ranges should be a non-empty dictionary.")
        return False

    logger.info("Metadata verification passed.")
    return True

def process_block(frame, x, y):
    """Process a block of pixels from the frame to get average color."""
    if y + 4 <= frame.shape[0] and x + 4 <= frame.shape[1]:
        block_color = frame[y:y + 4, x:x + 4].mean(axis=(0, 1))  # Average color in the block
        r, g, b = map(int, block_color)  # Convert to integers
        logger.debug(f"Average color for block ({x}, {y}): R={r}, G={g}, B={b}")
        return (r, g, b)
    logger.warning(f"Block out of frame bounds for ({x}, {y})")
    return None

def reconstruct_bytes_from_frame(frame, color_mapping, metadata):
    """Reconstruct original bytes from a video frame using color mapping."""
    original_bytes = bytearray()
    logger.debug(f"Decrypting frame with metadata: {metadata}")

    for byte_index, byte_info in metadata['byte_ranges'].items():
        if 'coordinates' in byte_info:
            x, y = byte_info['coordinates']
            block_color = process_block(frame, x, y)

            logger.debug(f"Block color at ({x}, {y}): {block_color}")

            if block_color:
                found = False
                for mapped_byte_index, color in color_mapping.items():
                    logger.debug(f"Checking mapped color: {color} against block color: {block_color}")
                    if (color['r'], color['g'], color['b']) == block_color:
                        original_bytes.append(int(mapped_byte_index))
                        found = True
                        logger.info(f"Found matching color for block at ({x}, {y}): {block_color} -> {mapped_byte_index}")
                        break
                
                if not found:
                    logger.warning(f"No matching color found for block at ({x}, {y}): {block_color}")
            else:
                logger.warning(f"Block color not computed at ({x}, {y})")

    logger.debug(f"Original bytes decrypted: {original_bytes}")
    return original_bytes

async def convert_video_to_file(video_path: str):
    """Convert a video to a file using metadata for reconstruction."""
    try:
        # Load video metadata
        if not os.path.exists(METADATA_PATH):
            logger.error(f"Metadata file not found at {METADATA_PATH}")
            return None
        
        with open(METADATA_PATH, 'r') as metadata_file:
            metadata = json.load(metadata_file)

        logger.info(f"Loaded metadata for video: {metadata}")

        # Verify metadata structure
        if not verify_metadata(metadata):
            logger.error("Metadata verification failed. Aborting processing.")
            return None

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Unable to open video file: {video_path}")
            return None

        # Load the color mapping
        if not os.path.exists(COLOR_MAPPING_PATH):
            logger.error(f"Color mapping file not found at {COLOR_MAPPING_PATH}")
            return None

        with open(COLOR_MAPPING_PATH, 'r') as mapping_file:
            color_mapping = json.load(mapping_file)

        original_bytes = bytearray()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {frame_count}")

        # Set the current position to frame 6
        cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
        futures = []
        ret, frame = cap.read()  # Read frame 6
        if not ret or frame is None:
            logger.error(f"Failed to read frame 6. Stopping reconstruction.")
        else:
            logger.info(f"Submitting frame 6 for processing")
            
            # Process blocks for frame 6
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                for block_key, block_info in metadata['byte_ranges'].items():
                    if block_info['frame'] == 6:  # Process only relevant blocks
                        logger.debug(f"Processing block {block_info['block']} for frame 6")
                        futures.append(executor.submit(reconstruct_bytes_from_frame, frame, color_mapping, metadata))

        for future in as_completed(futures):
            try:
                frame_bytes = future.result()
                logger.debug(f"Frame bytes received: {frame_bytes}, type: {type(frame_bytes)}")

                if isinstance(frame_bytes, bytearray):
                    original_bytes.extend(frame_bytes)
                else:
                    logger.error(f"Unexpected type for frame_bytes: {type(frame_bytes)}. Expected bytearray.")
            except Exception as e:
                logger.error(f"Error processing future: {str(e)}")

        cap.release()

        # Write the reconstructed byte array to a file
        with open(RECONSTRUCTED_FILE_PATH, "wb") as reconstructed_file:
            reconstructed_file.write(original_bytes)

        logger.info(f"Reconstructed file successfully saved to: {RECONSTRUCTED_FILE_PATH}")
        
        # Check the reconstructed file properties
        check_file_properties(metadata)

        return RECONSTRUCTED_FILE_PATH

    except Exception as e:
        logger.error(f"An error occurred during video to file conversion: {str(e)}")
        return None

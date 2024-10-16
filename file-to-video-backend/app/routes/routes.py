from loguru import logger
import os
import aiofiles
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from utils.utils import convert_video_to_file
from utils.conversion import convert_file_to_video

# Set up loguru logging configuration
logger.add("app.log", rotation="10 MB", level="INFO", format="{time} {level} {message}")

file_router = APIRouter()

# Ensure directories exist
os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("generated_videos", exist_ok=True)
os.makedirs("mappings", exist_ok=True)

# Route to upload and convert file to video
@file_router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Calculate the file size
        file.file.seek(0, os.SEEK_END)  # Move to the end of the file
        file_size = file.file.tell()  # Get the current position, which is the size
        file.file.seek(0)  # Reset the file pointer to the start for further processing
        
        # Log the file name and size
        logger.info(f"Uploaded file: {file.filename}, Size: {file_size} bytes")
        
        video_path = await convert_file_to_video(file)
        
        # Log successful video conversion
        logger.info(f"Successfully converted file to video: {video_path}")
        
        return {"video_url": video_path}
    except Exception as e:
        # Log the error with traceback
        logger.error(f"Error during file upload and conversion: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# Route to convert video back to file
@file_router.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Log the upload of the video
        logger.info(f"Uploaded video: {video.filename}")
        
        # Save the uploaded video to a temporary location
        temp_video_path = f"uploaded_files/{video.filename}"
        async with aiofiles.open(temp_video_path, "wb") as temp_video:
            content = await video.read()
            await temp_video.write(content)
        
        logger.info(f"Video saved temporarily at {temp_video_path}")
        
        # Pass the saved file path to your conversion function
        file_path = await convert_video_to_file(temp_video_path)
        
        # Log successful conversion back to file
        logger.info(f"Successfully converted video to file: {file_path}")
        
        return {"file_url": file_path}
    except Exception as e:
        # Log the error with traceback
        logger.error(f"Error during video upload and conversion: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

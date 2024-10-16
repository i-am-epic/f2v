from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import file_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include file-related routes
app.include_router(file_router)

@app.get("/")
async def root():
    return {"message": "File to Video API running!"}

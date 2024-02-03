from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .upload_endpoint import router as upload_router

app = FastAPI()

# Set up CORS middleware
origins = [
    # "http://localhost:8080",  # INPUT_REQUIRED {add any other origins as needed}
    "http://127.0.0.1:8080",  # INPUT_REQUIRED {add any other origins as needed}
    # Add other origins/ports your frontend may run on, or use ["*"] for open access
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows specific origins (you can use ["*"] for development purposes)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(upload_router, prefix="/upload")


@app.get("/")
async def root():
    return {"message": "PDF Parser v4 is online"}

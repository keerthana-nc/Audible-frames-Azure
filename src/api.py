"""
api.py -- FastAPI web application for Audible Frames.

ENDPOINTS
---------
  GET  /health    -> {"status": "ok"}
                     Liveness check -- Azure uses this to know the app is alive.

  POST /describe  -> accepts an image file upload, runs the pipeline,
                     returns a WAV audio file + text description in a header.

HOW TO RUN LOCALLY
------------------
  # Make sure your venv is active and .env is filled in, then:
  uvicorn src.api:app --reload
  # Open http://localhost:8000/docs in your browser to test it interactively.

PYDANTIC NOTE
-------------
FastAPI 0.100+ uses Pydantic v2. If you ever see errors about
"pydantic.v1" or "__fields__", something is using the old Pydantic v1 API.
Check that pydantic>=2.0 is installed (our smoke test verifies this).
"""

import os
import logging

# load_dotenv() reads the .env file and puts all its key=value pairs into
# os.environ, so all modules can access them with os.environ["KEY"].
# This must happen before any Azure client is created.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from src.pipeline import run_pipeline

# Set up basic logging so we can see what's happening in the terminal.
# In Phase 5 we'll upgrade this to Azure Application Insights.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Allowed image types ---
# Azure AI Vision supports: JPEG, PNG, BMP, GIF, TIFF, ICO, WEBP, and more.
# We restrict to the most common ones to keep validation simple.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/gif",
}

# Azure AI Vision Image Analysis has a 4MB file size limit.
MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024  # 4MB


# --- Create the FastAPI app ---
# The title and description show up in the auto-generated /docs page.
app = FastAPI(
    title="Audible Frames",
    description=(
        "Converts images into spoken audio descriptions for people who are "
        "blind or have low vision. "
        "Pipeline: Image -> Azure AI Vision -> GPT-5.4-mini -> Azure Speech -> Audio."
    ),
    version="0.2.0",
)


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns {"status": "ok"} when the server is running.
    Azure Container Apps pings this to decide if the app is healthy.
    If it returns anything other than 200 OK, Azure restarts the container.
    """
    return {"status": "ok"}


@app.post("/describe", tags=["Pipeline"])
async def describe_image(file: UploadFile = File(...)):
    """
    Accept an image, return a spoken description as a WAV audio file.

    **How to test this:**
    1. Open http://localhost:8000/docs in your browser
    2. Click "POST /describe" -> "Try it out"
    3. Upload any JPEG or PNG image
    4. Click "Execute" -- you'll get back an audio file you can download and play

    **What comes back:**
    - The response body is a WAV audio file (download and play it)
    - The X-Description header contains the text description (useful for debugging)
    - The X-Timing header contains per-step latency in JSON format

    **Limits:**
    - File must be JPEG, PNG, BMP, TIFF, or GIF
    - File must be under 4MB (Azure AI Vision limit)
    """

    # --- Input validation: check file type ---
    # file.content_type is set by the browser/client based on the file extension.
    # We reject anything that isn't an image type we support.
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,  # 415 = Unsupported Media Type
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # --- Read the file into memory ---
    # await file.read() reads the entire upload into a bytes object.
    # We do this once and reuse image_bytes, rather than reading the stream twice.
    image_bytes = await file.read()

    # --- Input validation: check file size ---
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,  # 413 = Content Too Large
            detail=(
                f"File is {len(image_bytes) / 1024 / 1024:.1f}MB. "
                f"Maximum allowed size is 4MB (Azure AI Vision limit)."
            ),
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Run the pipeline ---
    logger.info(f"Running pipeline on {file.filename} ({len(image_bytes)} bytes)")

    try:
        result = run_pipeline(image_bytes)
    except Exception as exc:
        # Log the full error for debugging, but return a clean message to the client.
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)}",
        )

    logger.info(
        f"Pipeline complete -- "
        f"vision={result['timing']['vision_ms']}ms, "
        f"gpt={result['timing']['captioner_ms']}ms, "
        f"speech={result['timing']['speech_ms']}ms, "
        f"total={result['timing']['total_ms']}ms"
    )

    # --- Return the audio file ---
    # We return the WAV bytes as the response body.
    # Headers carry the text description and timing so the caller gets both.
    import json
    return Response(
        content=result["audio_bytes"],
        media_type="audio/wav",
        headers={
            # X- prefix is the convention for custom HTTP headers.
            "X-Description": result["description"],
            "X-Timing": json.dumps(result["timing"]),
            # Content-Disposition tells the browser to download the file with this name.
            "Content-Disposition": "attachment; filename=description.wav",
        },
    )

"""
api.py — FastAPI web application for Audible Frames.

PURPOSE
-------
This is the entry point of the app. It exposes HTTP endpoints that clients
(a browser, a mobile app, a script) call to use the pipeline.

ENDPOINTS
---------
  GET  /health    → {"status": "ok"}
                    Used by Azure to check the app is alive. Implemented now.

  POST /describe  → accepts an image file, returns text + audio.
                    Implemented in Phase 2.

WHY FASTAPI?
------------
  - Auto-generates interactive API docs at /docs — great for demos and interviews
  - Async-ready: Azure API calls don't block while waiting for a response
  - Native Pydantic v2 integration: request/response data is validated automatically
  - Fast to write and easy to read

PYDANTIC NOTE
-------------
FastAPI 0.100+ uses Pydantic v2. If you ever see an error about
"pydantic.v1" or model_fields vs __fields__, it means something is trying
to use Pydantic v1. Check requirements.txt — pydantic must be 2.x.

RUNNING LOCALLY
---------------
  uvicorn src.api:app --reload
  # --reload: auto-restarts when you save a file (great for development)
  # Then open http://localhost:8000/docs in your browser
"""

from fastapi import FastAPI

# ── App initialization ────────────────────────────────────────────────────────
# Create the FastAPI application object. The title and description appear
# in the auto-generated /docs page (Swagger UI).
app = FastAPI(
    title="Audible Frames",
    description=(
        "Converts images into spoken audio descriptions for people who are "
        "blind or have low vision. "
        "Pipeline: Image → Azure AI Vision → GPT-5.4-mini → Azure Speech → Audio."
    ),
    version="0.1.0",
)


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check.

    Returns {"status": "ok"} when the server is running correctly.

    Azure Container Apps and load balancers call this endpoint periodically.
    If it stops returning 200 OK, Azure considers the app unhealthy and
    restarts it automatically.

    This is the one endpoint that works right now (Phase 1).
    All others are added in Phase 2.
    """
    return {"status": "ok"}


# ── /describe endpoint (Phase 2) ──────────────────────────────────────────────
# Phase 2 will add:
#
# from fastapi import UploadFile, File, HTTPException
# from fastapi.responses import Response
# from src.pipeline import run_pipeline
#
# @app.post("/describe", tags=["Pipeline"])
# async def describe_image(file: UploadFile = File(...)):
#     """
#     Accept an image, return a text description and audio file.
#
#     - **file**: The image to describe (JPEG or PNG, max 4MB)
#
#     Returns:
#     - JSON body with the text description
#     - Audio as a downloadable FLAC file attachment
#     """
#     # Read the uploaded image into memory.
#     image_bytes = await file.read()
#
#     # Run the full pipeline: Vision → GPT → Speech.
#     result = run_pipeline(image_bytes)
#
#     # Return audio as a downloadable file.
#     return Response(
#         content=result["audio_bytes"],
#         media_type="audio/flac",
#         headers={
#             "X-Description": result["description"],
#             "Content-Disposition": "attachment; filename=description.flac",
#         },
#     )

"""
api.py -- FastAPI web application for Audible Frames.

ENDPOINTS
---------
  GET  /          -> Serves the chat-style web UI (HTML page)
  GET  /health    -> {"status": "ok"}  liveness check for Azure
  POST /describe  -> accepts image upload, returns WAV audio + text description

HOW TO RUN LOCALLY
------------------
  uvicorn src.api:app --reload
  Open http://localhost:8000 in your browser.
"""

import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()  # must run before any Azure client is created

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, HTMLResponse

from src.pipeline import run_pipeline
from src.content_safety import ContentSafetyError
from src.telemetry import setup_telemetry, track_pipeline_request, track_safety_rejection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Set up Application Insights if connection string is configured.
# Safe to call even if the env var is missing -- it just skips silently.
setup_telemetry()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/gif", "image/webp",
}
MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024  # 4MB -- Azure AI Vision limit

app = FastAPI(
    title="Audible Frames",
    description="Converts images into spoken audio descriptions for accessibility.",
    version="0.2.0",
)


# =============================================================================
# Web UI
# =============================================================================

# The HTML for the chat-style interface.
# It's a single self-contained page: no external CSS files, no JS files.
# The JavaScript uses fetch() to call POST /describe, receives the WAV bytes
# as a Blob, creates a local URL from it, and plays it with an HTML5 <audio>
# element -- no file download required.
UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Audible Frames</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e8eaf0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }

    h1 {
      font-size: 1.8rem;
      font-weight: 700;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #6b7280;
      font-size: 0.9rem;
      margin-bottom: 40px;
    }

    /* --- Chat window --- */
    .chat-window {
      width: 100%;
      max-width: 680px;
      background: #1a1d27;
      border-radius: 16px;
      border: 1px solid #2d3148;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 420px;
    }

    .messages {
      flex: 1;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      overflow-y: auto;
    }

    /* A single chat bubble */
    .bubble {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 14px;
      line-height: 1.5;
      font-size: 0.95rem;
    }

    /* System messages (left side, grey) */
    .bubble.system {
      background: #252840;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
    }

    /* User messages (right side, blue) */
    .bubble.user {
      background: #1d4ed8;
      align-self: flex-end;
      border-bottom-right-radius: 4px;
    }

    /* Result bubble with description + audio player */
    .bubble.result {
      background: #1a2740;
      border: 1px solid #2d4a7a;
      align-self: flex-start;
      border-bottom-left-radius: 4px;
      width: 85%;
    }

    .bubble.result .label {
      font-size: 0.75rem;
      color: #60a5fa;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }

    .bubble.result .description-text {
      color: #e2e8f0;
      margin-bottom: 14px;
      line-height: 1.6;
    }

    audio {
      width: 100%;
      height: 36px;
      border-radius: 8px;
      accent-color: #60a5fa;
    }

    /* Error bubble */
    .bubble.error {
      background: #3b1a1a;
      border: 1px solid #7f1d1d;
      color: #fca5a5;
      align-self: flex-start;
    }

    /* Loading dots animation */
    .bubble.loading {
      background: #252840;
      align-self: flex-start;
      color: #6b7280;
    }

    .dots span {
      animation: blink 1.4s infinite;
      font-size: 1.4rem;
      line-height: 0;
    }
    .dots span:nth-child(2) { animation-delay: 0.2s; }
    .dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
      0%, 80%, 100% { opacity: 0; }
      40% { opacity: 1; }
    }

    /* --- Input bar at the bottom --- */
    .input-bar {
      padding: 16px;
      border-top: 1px solid #2d3148;
      display: flex;
      align-items: center;
      gap: 12px;
      background: #13151f;
    }

    /* Hidden real file input */
    #file-input { display: none; }

    /* Styled upload button */
    .upload-btn {
      flex-shrink: 0;
      padding: 10px 18px;
      background: #252840;
      border: 1px solid #3d4270;
      color: #c4c9e8;
      border-radius: 10px;
      cursor: pointer;
      font-size: 0.9rem;
      white-space: nowrap;
      transition: background 0.15s;
    }
    .upload-btn:hover { background: #2d3160; }

    /* Shows the selected file name */
    #file-name {
      flex: 1;
      font-size: 0.85rem;
      color: #6b7280;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* Send / describe button */
    #send-btn {
      flex-shrink: 0;
      padding: 10px 22px;
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      color: white;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 600;
      transition: opacity 0.15s;
    }
    #send-btn:disabled { opacity: 0.45; cursor: not-allowed; }
    #send-btn:not(:disabled):hover { opacity: 0.88; }
  </style>
</head>
<body>

  <h1>Audible Frames</h1>
  <p class="subtitle">Upload an image — get a spoken description back</p>

  <div class="chat-window">
    <div class="messages" id="messages">
      <!-- Initial greeting from the system -->
      <div class="bubble system">
        Hi! I turn images into audio descriptions for people who are blind or
        have low vision. Upload any photo and I'll describe what's in it.
      </div>
    </div>

    <div class="input-bar">
      <!-- Clicking the styled button triggers the hidden real file input -->
      <label class="upload-btn" for="file-input">Choose image</label>
      <input type="file" id="file-input" accept="image/*" />
      <span id="file-name">No file chosen</span>
      <button id="send-btn" disabled>Describe</button>
    </div>
  </div>

  <script>
    const fileInput  = document.getElementById("file-input");
    const fileNameEl = document.getElementById("file-name");
    const sendBtn    = document.getElementById("send-btn");
    const messages   = document.getElementById("messages");

    // Helper: add a chat bubble to the messages area and scroll to it.
    function addBubble(html, cssClass) {
      const div = document.createElement("div");
      div.className = "bubble " + cssClass;
      div.innerHTML = html;
      messages.appendChild(div);
      // Scroll to the bottom so the newest message is always visible.
      messages.scrollTop = messages.scrollHeight;
      return div;
    }

    // When the user picks a file, enable the Describe button and show the file name.
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        fileNameEl.textContent = file.name;
        sendBtn.disabled = false;
      }
    });

    // When the user clicks Describe:
    sendBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;

      // Show the user's image name as a "sent" bubble on the right.
      addBubble("Image: <strong>" + file.name + "</strong>", "user");

      // Show a loading indicator while we wait for Azure.
      const loadingBubble = addBubble(
        '<span class="dots"><span>.</span><span>.</span><span>.</span></span>',
        "loading"
      );

      // Disable the button while the request is in flight.
      sendBtn.disabled = true;

      try {
        // Build a FormData object with the image file.
        // This is what POST /describe expects (multipart/form-data).
        const form = new FormData();
        form.append("file", file);

        // Call POST /describe.
        const response = await fetch("/describe", { method: "POST", body: form });

        // Remove the loading indicator now that we have a response.
        loadingBubble.remove();

        if (!response.ok) {
          // The server returned an error -- show it in a red bubble.
          const errData = await response.json().catch(() => ({}));
          addBubble(
            "Error " + response.status + ": " + (errData.detail || response.statusText),
            "error"
          );
          return;
        }

        // The response body is WAV audio bytes.
        // We convert them to a Blob and create a local URL for the <audio> element.
        // This plays the audio right in the browser -- no download needed.
        const audioBlob = await response.blob();
        const audioUrl  = URL.createObjectURL(audioBlob);

        // Get the text description from the custom response header.
        const description = response.headers.get("X-Description") || "(no description)";

        // Build a result bubble with the text and an audio player.
        const bubble = addBubble(
          '<div class="label">Description</div>' +
          '<div class="description-text">' + description + '</div>' +
          '<audio controls autoplay src="' + audioUrl + '"></audio>',
          "result"
        );

      } catch (err) {
        loadingBubble.remove();
        addBubble("Network error: " + err.message, "error");
      } finally {
        // Re-enable the button so the user can try another image.
        sendBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the chat-style web interface."""
    return HTMLResponse(content=UI_HTML)


# =============================================================================
# API endpoints
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Liveness check. Returns {"status": "ok"} when the server is running.
    Azure Container Apps pings this to know the app is healthy.
    """
    return {"status": "ok"}


@app.post("/describe", tags=["Pipeline"])
async def describe_image(file: UploadFile = File(...)):
    """
    Accept an image file, run the full pipeline, return WAV audio.

    - Response body: WAV audio bytes (play directly with an HTML5 audio element)
    - X-Description header: the text description
    - X-Timing header: per-step latency in JSON (vision_ms, captioner_ms, speech_ms)
    """
    # Validate file type
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Upload a JPEG, PNG, BMP, TIFF, GIF, or WEBP image.",
        )

    # Read image bytes
    image_bytes = await file.read()

    # Validate size
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(image_bytes)/1024/1024:.1f}MB. Maximum is 4MB.",
        )

    logger.info(f"Processing {file.filename} ({len(image_bytes)} bytes)")

    try:
        result = run_pipeline(image_bytes)

    except ContentSafetyError as exc:
        # Image failed content safety check -- return 400, not 500.
        # 400 = "bad request" (the image itself is the problem, not our code).
        logger.warning(f"Content safety rejection for {file.filename}: {exc}")
        track_safety_rejection(file.filename, exc.categories)
        raise HTTPException(
            status_code=400,
            detail=f"Image rejected by content safety filter: {str(exc)}",
        )

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        track_pipeline_request(
            filename=file.filename,
            file_size_bytes=len(image_bytes),
            timing={},
            description_length=0,
            success=False,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    logger.info(
        f"Done -- safety={result['timing'].get('safety_ms', 0)}ms "
        f"vision={result['timing']['vision_ms']}ms "
        f"gpt={result['timing']['captioner_ms']}ms "
        f"speech={result['timing']['speech_ms']}ms "
        f"total={result['timing']['total_ms']}ms"
    )

    track_pipeline_request(
        filename=file.filename,
        file_size_bytes=len(image_bytes),
        timing=result["timing"],
        description_length=len(result["description"]),
        success=True,
    )

    return Response(
        content=result["audio_bytes"],
        media_type="audio/wav",
        headers={
            "X-Description": result["description"],
            "X-Timing": json.dumps(result["timing"]),
            # inline (not attachment) so the browser can play it without downloading
            "Content-Disposition": "inline; filename=description.wav",
        },
    )

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
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #141414;
      --surface:  #1e1e1e;
      --border:   #2a2a2a;
      --gold:     #c9a84c;
      --gold-dim: #7a6230;
      --text:     #f0ede6;
      --muted:    #6b6560;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
    }

    /* ── Header ── */
    header {
      width: 100%;
      max-width: 660px;
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 20px;
    }

    header h1 {
      font-size: 1.35rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      color: var(--gold);
    }

    header p {
      font-size: 0.82rem;
      color: var(--muted);
    }

    /* ── Chat window ── */
    .chat-window {
      width: 100%;
      max-width: 660px;
      height: 68vh;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* thin gold line at the very top */
    .chat-window::before {
      content: "";
      display: block;
      height: 2px;
      background: var(--gold);
      flex-shrink: 0;
    }

    /* ── Messages ── */
    .messages {
      flex: 1;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
    }

    .bubble {
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 3px;
      font-size: 0.92rem;
      line-height: 1.55;
    }

    /* system / AI — left */
    .bubble.system {
      background: var(--bg);
      border: 1px solid var(--border);
      align-self: flex-start;
      color: var(--text);
    }

    /* user — right, gold tint */
    .bubble.user {
      background: #1f1a0e;
      border: 1px solid var(--gold-dim);
      align-self: flex-end;
      color: var(--gold);
    }

    /* image thumbnail inside user bubble */
    .bubble.user img {
      display: block;
      max-width: 220px;
      max-height: 160px;
      object-fit: cover;
      border-radius: 2px;
      margin-top: 8px;
      border: 1px solid var(--gold-dim);
    }

    /* result bubble */
    .bubble.result {
      background: var(--bg);
      border: 1px solid var(--border);
      align-self: flex-start;
      width: 80%;
    }

    .bubble.result .label {
      font-size: 0.7rem;
      color: var(--gold);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }

    .bubble.result .desc {
      color: var(--text);
      margin-bottom: 14px;
      line-height: 1.65;
    }

    audio {
      width: 100%;
      height: 32px;
      accent-color: var(--gold);
    }

    /* error */
    .bubble.error {
      background: #1a0e0e;
      border: 1px solid #4a1a1a;
      color: #c47070;
      align-self: flex-start;
    }

    /* loading */
    .bubble.loading {
      background: var(--bg);
      border: 1px solid var(--border);
      align-self: flex-start;
      color: var(--muted);
      font-size: 1.2rem;
      letter-spacing: 0.15em;
    }

    @keyframes blink {
      0%, 80%, 100% { opacity: 0.15; }
      40%           { opacity: 1; }
    }
    .dots span { animation: blink 1.4s infinite; }
    .dots span:nth-child(2) { animation-delay: 0.2s; }
    .dots span:nth-child(3) { animation-delay: 0.4s; }

    /* ── Input bar ── */
    .input-bar {
      padding: 14px 16px;
      border-top: 1px solid var(--border);
      background: var(--bg);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    #file-input { display: none; }

    .upload-btn {
      flex-shrink: 0;
      padding: 8px 14px;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 3px;
      cursor: pointer;
      font-size: 0.83rem;
      white-space: nowrap;
      transition: border-color 0.12s, color 0.12s;
    }
    .upload-btn:hover { border-color: var(--gold); color: var(--gold); }

    #file-name {
      flex: 1;
      font-size: 0.82rem;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    #send-btn {
      flex-shrink: 0;
      padding: 8px 20px;
      background: var(--gold);
      color: #0e0c07;
      border: none;
      border-radius: 3px;
      cursor: pointer;
      font-size: 0.88rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      transition: opacity 0.12s;
    }
    #send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
    #send-btn:not(:disabled):hover { opacity: 0.82; }
  </style>
</head>
<body>

  <header>
    <h1>Audible Frames</h1>
    <p>image → spoken description</p>
  </header>

  <div class="chat-window">
    <div class="messages" id="messages">
      <div class="bubble system">
        Send me any image and I'll describe what's in it as audio — built for
        people who are blind or have low vision.
      </div>
    </div>

    <div class="input-bar">
      <label class="upload-btn" for="file-input">+ image</label>
      <input type="file" id="file-input" accept="image/*" />
      <span id="file-name">no file chosen</span>
      <button id="send-btn" disabled>Describe</button>
    </div>
  </div>

  <script>
    const fileInput  = document.getElementById("file-input");
    const fileNameEl = document.getElementById("file-name");
    const sendBtn    = document.getElementById("send-btn");
    const messages   = document.getElementById("messages");

    function addBubble(html, cssClass) {
      const div = document.createElement("div");
      div.className = "bubble " + cssClass;
      div.innerHTML = html;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
      return div;
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      fileNameEl.textContent = file.name;
      sendBtn.disabled = false;
    });

    sendBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;

      // Show image thumbnail in user bubble
      const reader = new FileReader();
      reader.onload = (e) => {
        addBubble(
          "<strong>" + file.name + "</strong>" +
          '<img src="' + e.target.result + '" alt="uploaded image" />',
          "user"
        );
      };
      reader.readAsDataURL(file);

      const loadingBubble = addBubble(
        '<span class="dots"><span>.</span><span>.</span><span>.</span></span>',
        "loading"
      );
      sendBtn.disabled = true;

      try {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/describe", { method: "POST", body: form });
        loadingBubble.remove();

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          addBubble("Error " + response.status + ": " + (errData.detail || response.statusText), "error");
          return;
        }

        const audioBlob = await response.blob();
        const audioUrl  = URL.createObjectURL(audioBlob);
        const description = response.headers.get("X-Description") || "";

        addBubble(
          '<div class="label">Description</div>' +
          '<div class="desc">' + description + '</div>' +
          '<audio controls autoplay src="' + audioUrl + '"></audio>',
          "result"
        );

      } catch (err) {
        loadingBubble.remove();
        addBubble("Network error: " + err.message, "error");
      } finally {
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

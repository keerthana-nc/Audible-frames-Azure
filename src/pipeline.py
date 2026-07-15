"""
pipeline.py -- Orchestrates the full image -> text -> audio pipeline.

PURPOSE
-------
This is the "traffic controller." It calls the three Azure services in order
and passes each output into the next step:

  Step 1: VisionClient.analyze()   -- image bytes  -> captions + OCR text
  Step 2: Captioner.describe()     -- captions     -> fluent text description
  Step 3: SpeechClient.synthesize()-- description  -> audio bytes (WAV)

TIMING
------
Each step is timed individually so Phase 5 (observability) can track where
latency lives. We return timing data alongside results so the API can log it.

WHY A SEPARATE FILE?
--------------------
Without this, api.py would need to know about all three Azure services and
would become one huge, hard-to-test file. Keeping orchestration here means:
  - We can test the pipeline logic independently of the web server
  - Each Azure service can be swapped without touching api.py
  - Timing and cleanup logic lives in one place
"""

import time

from src.vision import VisionClient
from src.captioner import Captioner
from src.speech import SpeechClient
from src.content_safety import ContentSafetyChecker, ContentSafetyError


def run_pipeline(image_bytes: bytes) -> dict:
    """
    Run the full image -> text -> audio pipeline.

    Args:
        image_bytes: Raw bytes of the input image (JPEG, PNG, BMP, GIF, TIFF).
                     Must be under 4MB (Azure AI Vision limit).

    Returns:
        {
            "description": "A developer sits at a desk focused on coding...",
            "audio_bytes": b"...raw WAV audio data...",
            "timing": {
                "vision_ms":    320,   # time spent in Azure AI Vision
                "captioner_ms": 1100,  # time spent in GPT
                "speech_ms":    450,   # time spent in Azure Speech
                "total_ms":     1870   # total wall-clock time
            }
        }

    Raises:
        RuntimeError: If any Azure service call fails. The error message will
                      say which step failed and why.
    """
    timings = {}

    # --- Step 0: Azure AI Content Safety ---
    # Check the image for harmful content BEFORE sending it to any AI model.
    # ContentSafetyError is raised here if the image is rejected -- the caller
    # (api.py) catches it and returns a 400 error to the user.
    t0 = time.monotonic()
    checker = ContentSafetyChecker()
    checker.check_image(image_bytes)   # raises ContentSafetyError if unsafe
    timings["safety_ms"] = round((time.monotonic() - t0) * 1000)

    # --- Step 1: Azure AI Vision ---
    # Send the image to Azure and get back what's in it.
    t0 = time.monotonic()
    # time.monotonic() is a clock that never goes backwards -- safe for measuring durations.
    # (regular time.time() can jump if the system clock changes)

    vision_client = VisionClient()
    vision_output = vision_client.analyze(image_bytes)
    timings["vision_ms"] = round((time.monotonic() - t0) * 1000)
    # * 1000 converts seconds to milliseconds. round() removes fractional ms.

    # --- Step 2: Azure OpenAI GPT ---
    # Turn the Vision output into a fluent human description.
    t0 = time.monotonic()

    captioner = Captioner()
    description = captioner.describe(vision_output)
    timings["captioner_ms"] = round((time.monotonic() - t0) * 1000)

    # --- Step 3: Azure AI Speech ---
    # Convert the description to audio.
    t0 = time.monotonic()

    speech_client = SpeechClient()
    audio_bytes = speech_client.synthesize(description)
    timings["speech_ms"] = round((time.monotonic() - t0) * 1000)

    # Total is the sum of all steps including safety check.
    timings["total_ms"] = (
        timings["safety_ms"]
        + timings["vision_ms"]
        + timings["captioner_ms"]
        + timings["speech_ms"]
    )

    return {
        "description": description,
        "audio_bytes": audio_bytes,
        "timing": timings,
    }

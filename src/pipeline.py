"""
pipeline.py — Orchestrates the full image → text → audio pipeline.

PURPOSE
-------
This is the "traffic controller" of the app. It calls the three Azure services
in order and passes the output of each step into the next:

  Step 1: vision.py   — image bytes  → dense captions + OCR text
  Step 2: captioner.py — captions    → fluent text description
  Step 3: speech.py   — description  → audio bytes (FLAC/MP3)

The pipeline returns both the text description AND the audio, so the API
can give clients both (useful for testing and for users who want the text too).

TIMING
------
Each step is timed individually so Phase 5 (observability) can track where
latency lives: is the bottleneck in Vision? GPT? Speech?

CLEANUP
-------
Any ephemeral objects created during a run (temporary image files, intermediate
audio files) are deleted in a `finally` block. This runs even if the pipeline
crashes partway through, so no run ever leaves billable leftover files.

WHY A SEPARATE pipeline.py?
----------------------------
Without this, api.py would have to know about all three Azure services — it
would become one giant file that's hard to test. Keeping orchestration in its
own file means:
  - We can test the pipeline logic without running the web server
  - Each service can be swapped out (e.g. different TTS engine) without
    touching the API code

PHASE 1 STATUS
--------------
Documented stub — run_pipeline() implemented in Phase 2.
"""

# Phase 2 implementation will go here.
#
# from src.vision import VisionClient
# from src.captioner import Captioner
# from src.speech import SpeechClient
# import time
#
# def run_pipeline(image_bytes: bytes) -> dict:
#     '''
#     Run the full image → text → audio pipeline.
#
#     Args:
#         image_bytes: Raw bytes of the input image (JPEG, PNG, etc.)
#
#     Returns:
#         {
#             "description": "A fluent text description of the image.",
#             "audio_bytes": b"...raw FLAC audio...",
#             "timing": {
#                 "vision_ms": 320,
#                 "captioner_ms": 1100,
#                 "speech_ms": 450,
#                 "total_ms": 1870
#             }
#         }
#     '''
#     timings = {}
#     temp_files = []   # track any temp files so we can clean up in finally
#
#     try:
#         # Step 1: Vision
#         t0 = time.monotonic()
#         vision_output = VisionClient().analyze(image_bytes)
#         timings["vision_ms"] = round((time.monotonic() - t0) * 1000)
#
#         # Step 2: Caption
#         t0 = time.monotonic()
#         description = Captioner().describe(vision_output)
#         timings["captioner_ms"] = round((time.monotonic() - t0) * 1000)
#
#         # Step 3: Speech
#         t0 = time.monotonic()
#         audio_bytes = SpeechClient().synthesize(description)
#         timings["speech_ms"] = round((time.monotonic() - t0) * 1000)
#
#         timings["total_ms"] = sum(timings.values())
#
#         return {"description": description, "audio_bytes": audio_bytes, "timing": timings}
#
#     finally:
#         # Clean up any temp files this run created.
#         # This block runs even if an exception was raised above.
#         for path in temp_files:
#             if path.exists():
#                 path.unlink()
#                 # .unlink() is pathlib's way of deleting a file.

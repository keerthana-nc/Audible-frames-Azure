"""
speech.py — Azure AI Speech (Text-to-Speech) integration.

PURPOSE
-------
This module takes a text description (produced by captioner.py) and converts
it into audio using Azure's neural TTS voices. The result is an audio file
that can be played directly by a screen-reader user or embedded in a web app.

OUTPUT FORMAT
-------------
FLAC — lossless audio compression. Good choice for accessibility because:
  - No quality loss (unlike MP3)
  - Smaller than raw WAV
  - Widely supported by screen readers and media players

We can also produce MP3 if the client prefers smaller files.

VOICE
-----
Default: en-US-AriaNeural — a natural, clear English voice rated highly for
accessibility use cases. Azure offers 400+ voices across 140 languages;
the voice can be made configurable via env var for future internationalization.

WHY NEURAL TTS?
---------------
Older TTS systems sound robotic and are tiring to listen to for long descriptions.
Azure's neural voices are trained on human speech and sound significantly more
natural — important for an assistive tech use case where users may listen to
dozens of descriptions per day.

CLEANUP
-------
Any temporary audio files written to disk during a run will be deleted in a
`finally` block in pipeline.py. This ensures no run leaves audio files behind
that would accumulate and (in theory) incur storage costs.

AZURE SDK
---------
Package: azure-cognitiveservices-speech
Windows note: pip installs the correct Windows-specific wheel automatically.

PHASE 1 STATUS
--------------
Documented stub — SpeechClient class implemented in Phase 2.
"""

# Phase 2 implementation will go here.
#
# class SpeechClient:
#     def __init__(self):
#         # Create a SpeechConfig from AZURE_SPEECH_KEY and AZURE_SPEECH_REGION.
#         # Set the output format to FLAC (or MP3 based on config).
#         # Set the voice to en-US-AriaNeural (or from env var).
#         ...
#
#     def synthesize(self, text: str) -> bytes:
#         # Send text to Azure Speech → get audio bytes back.
#         # Any temp files are cleaned up in a finally block.
#         # Returns: raw audio bytes (FLAC or MP3).
#         ...

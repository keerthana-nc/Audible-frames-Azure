"""
vision.py — Azure AI Vision (Image Analysis) integration.

PURPOSE
-------
This module talks to Azure AI Vision to extract two things from an image:

  1. Dense captions — natural-language descriptions of what's in the image,
     region by region. Example: "a golden retriever sitting on a park bench"
     or "text on a sign reading 'OPEN'".

  2. OCR text — any printed or handwritten text found in the image, read out
     verbatim.

WHY THIS EXISTS
---------------
Raw image bytes don't mean anything to a language model. Azure AI Vision
translates pixels into structured text that GPT (in captioner.py) can then
turn into a fluent, useful description for a screen-reader user.

Dense captions give us scene-level understanding, not just a flat list of
objects — so the downstream GPT description is much richer.

AZURE SDK
---------
Package: azure-ai-vision-imageanalysis (v1.0+)
Docs: https://learn.microsoft.com/azure/ai-services/computer-vision/how-to/image-analysis

PHASE 1 STATUS
--------------
This file is a documented stub. The VisionClient class is implemented in Phase 2.
"""

# Phase 2 implementation will go here.
#
# What VisionClient will look like:
#
# class VisionClient:
#     def __init__(self):
#         # Read credentials from environment variables (never hardcode keys).
#         # Connect to Azure AI Vision using the azure-ai-vision-imageanalysis SDK.
#         ...
#
#     def analyze(self, image_bytes: bytes) -> dict:
#         # Send image to Azure → get dense captions + OCR text back.
#         # Returns: {"captions": [...], "ocr_text": "..."}
#         ...

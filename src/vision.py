"""
vision.py -- Azure AI Vision (Image Analysis) integration.

PURPOSE
-------
Sends an image to Azure AI Vision and gets back:
  1. Dense captions  -- natural-language descriptions of what's in the image
                        e.g. ["a person sitting at a desk", "a laptop with code on screen"]
  2. OCR text        -- any printed/handwritten text found in the image
                        e.g. "def main():"

WHY TWO THINGS?
---------------
Dense captions tell us WHAT'S IN the scene (objects, people, actions).
OCR text tells us WHAT'S WRITTEN in the scene (signs, code, labels).
Both together give GPT (in captioner.py) the full picture to write a
rich, accurate description for a screen-reader user.

AZURE SDK
---------
Package: azure-ai-vision-imageanalysis
Docs: https://learn.microsoft.com/azure/ai-services/computer-vision/
"""

import os

# ImageAnalysisClient is the main class we use to talk to Azure AI Vision.
from azure.ai.vision.imageanalysis import ImageAnalysisClient

# VisualFeatures is an enum that tells Azure WHAT to analyze in the image.
# We request DENSE_CAPTIONS (scene descriptions) and READ (OCR text).
from azure.ai.vision.imageanalysis.models import VisualFeatures

# AzureKeyCredential wraps our API key in a format the SDK expects.
from azure.core.credentials import AzureKeyCredential


class VisionClient:
    """
    Wraps Azure AI Vision to extract captions and text from an image.

    Usage:
        client = VisionClient()
        result = client.analyze(image_bytes)
        # result = {"captions": ["a cat on a sofa"], "ocr_text": "OPEN"}
    """

    def __init__(self):
        """
        Initialize the Azure AI Vision client using credentials from environment variables.

        We read from env vars (not hardcode) so:
          - Credentials never end up in code or git history
          - Switching to a different Azure resource is a .env change, not a code change
        """
        endpoint = os.environ["AZURE_VISION_ENDPOINT"]
        key = os.environ["AZURE_VISION_KEY"]

        # Create the Azure AI Vision client.
        # AzureKeyCredential(key) wraps the key string in a credential object the SDK requires.
        self.client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def analyze(self, image_bytes: bytes) -> dict:
        """
        Send raw image bytes to Azure AI Vision and return structured results.

        Args:
            image_bytes: The image to analyze, as raw bytes (JPEG, PNG, BMP, GIF, TIFF).
                         Max size: 4MB (Azure limit for Image Analysis).

        Returns:
            {
                "captions": ["a cat sitting on a sofa", "a window with sunlight"],
                "ocr_text": "OPEN DAILY 9am-5pm"   # empty string if no text found
            }
        """
        # Call Azure AI Vision with the image data.
        # visual_features tells Azure which analyses to run.
        # DENSE_CAPTIONS: describe the overall scene and prominent regions in natural language.
        # READ: OCR -- extract any printed or handwritten text from the image.
        result = self.client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ],
        )

        # --- Extract dense captions ---
        # result.dense_captions.list is a list of Caption objects.
        # Each has a .text (the description) and a .confidence (0.0 to 1.0).
        # We only keep captions with decent confidence to avoid noise.
        captions = []
        if result.dense_captions and result.dense_captions.list:
            for cap in result.dense_captions.list:
                # confidence threshold: only keep captions Azure is reasonably sure about
                if cap.confidence >= 0.5:
                    captions.append(cap.text)

        # --- Extract OCR text ---
        # result.read.blocks is a list of text regions in the image.
        # Each block has lines, and each line has words.
        # We join all lines into a single string.
        ocr_lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    ocr_lines.append(line.text)

        ocr_text = " ".join(ocr_lines)

        return {
            "captions": captions,
            "ocr_text": ocr_text,
        }

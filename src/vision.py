"""
vision.py -- Azure AI Vision (Image Analysis) integration.

PURPOSE
-------
Sends an image to Azure AI Vision and returns:
  1. Caption  -- one natural-language sentence describing the whole scene
                 e.g. "a developer sitting at a desk working on a laptop"
  2. Tags     -- list of objects/concepts detected with high confidence
                 e.g. ["laptop", "person", "coffee", "indoor"]
  3. OCR text -- any printed or handwritten text found in the image
                 e.g. "def main():"

WHY CAPTION + TAGS (not DenseCaptions)?
----------------------------------------
DenseCaptions describes multiple regions of the image but is only available
in a handful of Azure regions. CAPTION (single image-level caption) + TAGS
are available in ALL regions including East US 2, and together give GPT
rich context to write a good description:
  - Caption tells GPT the overall scene
  - Tags add detail about individual objects
  - OCR adds any text visible in the image

AZURE SDK
---------
Package: azure-ai-vision-imageanalysis
Docs: https://learn.microsoft.com/azure/ai-services/computer-vision/
"""

import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


class VisionClient:
    """
    Wraps Azure AI Vision to extract a caption, tags, and text from an image.

    Usage:
        client = VisionClient()
        result = client.analyze(image_bytes)
        # result = {
        #   "captions": ["a developer at a desk", "Also detected: laptop, coffee, indoor"],
        #   "ocr_text": "def main():"
        # }
    """

    def __init__(self):
        """
        Initialize the Azure AI Vision client from environment variables.
        Credentials are never hardcoded -- always read from .env.
        """
        self.client = ImageAnalysisClient(
            endpoint=os.environ["AZURE_VISION_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_VISION_KEY"]),
        )

    def analyze(self, image_bytes: bytes) -> dict:
        """
        Send image bytes to Azure AI Vision and return structured results.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, BMP, GIF, TIFF). Max 4MB.

        Returns:
            {
                "captions": [
                    "a developer sitting at a desk with a laptop",  # from CAPTION
                    "Also detected: laptop, person, coffee, indoor" # from TAGS
                ],
                "ocr_text": "def main():"  # from READ (empty string if no text)
            }
        """
        # Request three visual features from Azure:
        #   CAPTION  -- one sentence describing the whole image
        #   TAGS     -- list of objects/concepts detected (car, sky, person, etc.)
        #   READ     -- OCR: any text visible in the image
        #
        # Note: DENSE_CAPTIONS is NOT used here because it's only available in
        # select Azure regions. CAPTION + TAGS work everywhere including East US 2.
        result = self.client.analyze(
            image_data=image_bytes,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.READ,
            ],
        )

        context_pieces = []

        # --- Extract the main caption ---
        # result.caption is a single Caption object with .text and .confidence.
        # confidence is between 0.0 and 1.0 -- we only use it if Azure is reasonably sure.
        if result.caption and result.caption.confidence >= 0.4:
            context_pieces.append(result.caption.text)

        # --- Extract tags ---
        # Tags are individual objects, concepts, or scene attributes Azure detected.
        # We filter by confidence (>= 0.7) and take the top 8 to keep the prompt concise.
        if result.tags and result.tags.list:
            high_conf_tags = [
                t.name
                for t in result.tags.list
                if t.confidence >= 0.7
            ][:8]
            if high_conf_tags:
                # Format as a supplementary line for GPT to incorporate.
                context_pieces.append("Also detected: " + ", ".join(high_conf_tags))

        # --- Extract OCR text ---
        # result.read.blocks is a list of text regions; each has lines.
        ocr_lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    ocr_lines.append(line.text)

        return {
            "captions": context_pieces,
            "ocr_text": " ".join(ocr_lines),
        }

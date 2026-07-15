"""
content_safety.py -- Azure AI Content Safety guardrail.

PURPOSE
-------
Before an image goes through Vision + GPT + Speech, we check whether it
contains harmful content. If it does, we reject it immediately with a clear
error -- the image never reaches the AI models.

This protects against:
  - Users accidentally uploading harmful images
  - Bad actors trying to get the AI to describe violent/sexual content

HOW IT WORKS
------------
Azure AI Content Safety scans the image and returns a severity score (0-6)
for four categories:

  Hate      -- hateful symbols, gestures, text in image
  SelfHarm  -- depictions of self-harm
  Sexual    -- sexually explicit content
  Violence  -- graphic violence

We reject the image if ANY category scores >= SEVERITY_THRESHOLD (default 2).
  0-1 = safe to process
  2+  = reject

ENVIRONMENT VARIABLES REQUIRED
-------------------------------
  AZURE_CONTENT_SAFETY_ENDPOINT  -- e.g. https://your-resource.cognitiveservices.azure.com/
  AZURE_CONTENT_SAFETY_KEY       -- your Content Safety API key
  (Both are in your .env file)
"""

import os

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


# Severity threshold: reject if any category is >= this value.
# 0 = no harmful content detected, 6 = most severe.
# 2 is a conservative threshold -- flags mild content.
# Raise to 4 if you want to allow more edge cases through.
SEVERITY_THRESHOLD = 2


class ContentSafetyError(Exception):
    """
    Raised when an image fails the content safety check.

    Attributes:
        categories: dict of {category_name: severity_score} for all four
                    categories, so the caller can log what was found.
    """
    def __init__(self, message: str, categories: dict):
        super().__init__(message)
        self.categories = categories


class ContentSafetyChecker:
    """
    Wraps the Azure AI Content Safety client.

    Creates one Azure client on init and reuses it for all calls -- creating
    clients is expensive (network connections), so we do it once.

    Usage:
        checker = ContentSafetyChecker()
        checker.check_image(image_bytes)   # raises ContentSafetyError if unsafe

    Or in a try/except:
        try:
            checker.check_image(image_bytes)
        except ContentSafetyError as e:
            return {"error": str(e), "categories": e.categories}
    """

    def __init__(self):
        endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "").strip()
        key      = os.environ.get("AZURE_CONTENT_SAFETY_KEY", "").strip()

        if not endpoint or not key:
            raise ValueError(
                "AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY "
                "must be set in your .env file."
            )

        self.client = ContentSafetyClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def check_image(self, image_bytes: bytes) -> dict:
        """
        Run the image through Azure AI Content Safety.

        Args:
            image_bytes: Raw bytes of the image to check (JPEG, PNG, GIF, BMP).
                         Maximum 2MB for Content Safety (smaller than Vision's 4MB limit).

        Returns:
            dict of {category: severity_score} -- all scores were safe (0 or 1).
            Example: {"Hate": 0, "SelfHarm": 0, "Sexual": 0, "Violence": 0}

        Raises:
            ContentSafetyError: If any category score >= SEVERITY_THRESHOLD.
            RuntimeError: If the Azure API call itself fails.
        """
        # Content Safety expects the raw image bytes directly
        image_data = ImageData(content=image_bytes)
        request    = AnalyzeImageOptions(image=image_data)

        try:
            response = self.client.analyze_image(request)
        except HttpResponseError as exc:
            # Azure API returned an HTTP error (auth failure, quota exceeded, etc.)
            raise RuntimeError(
                f"Content Safety API call failed: {exc.message}"
            ) from exc

        # Collect severity scores for each of the four categories
        scores = {
            result.category.value: result.severity
            for result in response.categories_analysis
        }

        # Find any categories that exceed the threshold
        violations = {
            cat: sev
            for cat, sev in scores.items()
            if sev >= SEVERITY_THRESHOLD
        }

        if violations:
            violation_str = ", ".join(
                f"{cat} (severity {sev}/6)"
                for cat, sev in violations.items()
            )
            raise ContentSafetyError(
                f"Image rejected: {violation_str}. "
                f"This image cannot be processed.",
                categories=scores,
            )

        # All scores were safe -- return them for logging purposes
        return scores

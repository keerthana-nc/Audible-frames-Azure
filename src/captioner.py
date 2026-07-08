"""
captioner.py -- Azure OpenAI GPT integration for scene description.

PURPOSE
-------
Takes structured output from vision.py (caption + tags + OCR text) and asks
GPT to write a single, fluent sentence or two describing the image -- the kind
a screen-reader user would actually want to hear.

WHY GPT ON TOP OF VISION?
--------------------------
Azure AI Vision gives us accurate but mechanical output:
  ["a developer at a desk", "Also detected: laptop, coffee, indoor"]

GPT synthesizes this into something human:
  "A developer sits at a desk working on code, with a coffee mug nearby."

API NOTE -- Responses API
--------------------------
GPT-5.4-mini uses the new Responses API (not the older Chat Completions API).
The key differences:
  Old (Chat Completions): client.chat.completions.create(messages=[...])
  New (Responses API):    client.responses.create(input=[...])

The parameter name changed from `messages` to `input`.
The response text is at: response.output_text  (convenience property)

AZURE SDK
---------
Package: openai (latest, upgraded from 1.57.0)
API version env var: AZURE_OPENAI_API_VERSION (must be 2025-03-01-preview or later)
"""

import os

from openai import AzureOpenAI


class Captioner:
    """
    Calls Azure OpenAI GPT to generate a fluent image description
    from the structured output produced by VisionClient.

    Usage:
        captioner = Captioner()
        description = captioner.describe({"captions": [...], "ocr_text": "..."})
        # -> "A developer sits at a desk focused on coding..."
    """

    def __init__(self):
        """
        Initialize the AzureOpenAI client from environment variables.

        Required env vars:
          AZURE_OPENAI_ENDPOINT         -- your Azure OpenAI resource URL
          AZURE_OPENAI_API_KEY          -- your API key
          AZURE_OPENAI_API_VERSION      -- must be 2025-03-01-preview or later
                                          (earlier versions don't support the Responses API)
          AZURE_OPENAI_DEPLOYMENT_NAME  -- the name you gave the model in Foundry
        """
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    def describe(self, vision_output: dict) -> str:
        """
        Generate a fluent, accessibility-focused image description using GPT.

        Args:
            vision_output: Dict from VisionClient.analyze():
                {
                    "captions": ["a person at a desk", "Also detected: laptop, coffee"],
                    "ocr_text": "def main():"
                }

        Returns:
            A natural-language description suitable for reading aloud.
        """
        # Build the context string we send to GPT as the user message.
        context_parts = []

        captions = vision_output.get("captions", [])
        if captions:
            context_parts.append("Visual elements: " + "; ".join(captions))

        ocr_text = vision_output.get("ocr_text", "").strip()
        if ocr_text:
            context_parts.append(f"Text visible in the image: {ocr_text}")

        if not context_parts:
            context_parts.append("No visual elements or text were detected.")

        context = "\n".join(context_parts)

        # --- Call GPT using the Responses API ---
        # GPT-5.4-mini uses the Responses API. Key difference from Chat Completions:
        #   - Parameter is `input=` (not `messages=`)
        #   - Response text is at `response.output_text`
        #
        # The `input` list works the same way as `messages` -- each item has a
        # `role` ("system" or "user") and `content` (the text).
        response = self.client.responses.create(
            model=self.deployment,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an accessibility assistant helping blind and low-vision users "
                        "understand images through audio descriptions. "
                        "Given structured image analysis data, write a single fluent description "
                        "in 2-3 sentences. "
                        "Be specific. Include any text visible in the image. "
                        "Write as if speaking to someone who cannot see the image. "
                        "Do not say 'The image shows' -- just describe what is there."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Describe this image based on the following analysis:\n\n{context}"
                    ),
                },
            ],
            max_output_tokens=300,  # renamed from max_tokens in the Responses API
            temperature=0.7,
        )

        # response.output_text is a convenience property that returns the
        # full text of the first (and usually only) output item.
        return response.output_text.strip()

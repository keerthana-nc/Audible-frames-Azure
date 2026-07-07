"""
captioner.py -- Azure OpenAI GPT integration for scene description.

PURPOSE
-------
Takes the structured output from vision.py (captions + OCR text) and asks
GPT to write a single, fluent sentence or paragraph describing the image --
the kind a screen-reader user would actually want to hear.

WHY GPT ON TOP OF VISION?
--------------------------
Azure AI Vision gives us accurate but mechanical output:
  ["a person sitting at a desk", "a laptop with code on screen", "a coffee mug"]

GPT synthesizes this into something human:
  "A developer sits at a desk working on code, with a coffee mug nearby."

That's the difference between a list of labels and a real description.

MODEL
-----
Read from AZURE_OPENAI_DEPLOYMENT_NAME env var.
  Primary:    gpt-5.4-mini   (East US 2, low latency, multimodal)
  Fallback 1: gpt-4o-mini
  Fallback 2: gpt-4o
Switching is a one-line change in .env -- no code rewrite needed.

AZURE SDK
---------
Package: openai (v1.x)
We use AzureOpenAI() -- the modern client for Azure endpoints.
NOT the deprecated openai.ChatCompletion.create() style.
"""

import os

# AzureOpenAI is the client for calling GPT via an Azure endpoint.
# It's part of the openai package (v1.x) -- same package as regular OpenAI,
# but configured with an Azure endpoint and key instead of an OpenAI key.
from openai import AzureOpenAI


class Captioner:
    """
    Calls Azure OpenAI GPT to generate a fluent image description
    from the structured output produced by VisionClient.

    Usage:
        captioner = Captioner()
        description = captioner.describe({"captions": [...], "ocr_text": "..."})
        # description = "A developer sits at a desk working on Python code..."
    """

    def __init__(self):
        """
        Initialize the AzureOpenAI client with credentials from environment variables.

        Three env vars are needed:
          AZURE_OPENAI_ENDPOINT    -- your Azure OpenAI resource URL
          AZURE_OPENAI_API_KEY     -- your API key
          AZURE_OPENAI_API_VERSION -- the API version string (controls which features are available)
          AZURE_OPENAI_DEPLOYMENT_NAME -- the name you gave the model when you deployed it in Foundry
        """
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )

        # The deployment name is the name YOU chose in Foundry -- not the model name.
        # This is what lets us switch models by changing one env var.
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    def describe(self, vision_output: dict) -> str:
        """
        Generate a fluent, accessibility-focused image description using GPT.

        Args:
            vision_output: The dict returned by VisionClient.analyze():
                {
                    "captions": ["a person at a desk", "a laptop"],
                    "ocr_text": "def main():"
                }

        Returns:
            A natural-language description suitable for a screen-reader user.
            Example: "A developer sits at a desk focused on coding. Their laptop
                      displays Python code including a main function definition."
        """
        # --- Build the context string we'll send to GPT ---
        # We format the Vision output into a readable block so GPT understands
        # exactly what was detected in the image.
        context_parts = []

        captions = vision_output.get("captions", [])
        if captions:
            # Join multiple captions with semicolons so GPT sees them as a list.
            context_parts.append("Visual elements: " + "; ".join(captions))

        ocr_text = vision_output.get("ocr_text", "").strip()
        if ocr_text:
            context_parts.append(f"Text visible in the image: {ocr_text}")

        if not context_parts:
            # Vision found nothing -- give GPT something to work with rather than failing.
            context_parts.append("No visual elements or text were detected in this image.")

        context = "\n".join(context_parts)

        # --- Call GPT ---
        # messages is a list of "turns" in the conversation.
        # "system" sets GPT's role and behavior.
        # "user" is the actual request.
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an accessibility assistant helping blind and low-vision users "
                        "understand images through audio descriptions. "
                        "Given structured image analysis data, write a single fluent description "
                        "in 2-4 sentences. "
                        "Be specific and descriptive. Include any text visible in the image. "
                        "Write as if speaking directly to someone who cannot see the image. "
                        "Do not say 'The image shows' -- just describe what is there."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Please describe this image based on the following analysis:\n\n{context}"
                    ),
                },
            ],
            max_tokens=300,   # enough for 2-4 sentences; keeps cost low
            temperature=0.7,  # slight creativity for natural-sounding language (0 = robotic, 1 = creative)
        )

        # response.choices[0].message.content is the text GPT generated.
        # .strip() removes any leading/trailing whitespace.
        return response.choices[0].message.content.strip()

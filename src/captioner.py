"""
captioner.py — Azure OpenAI GPT integration for scene description.

PURPOSE
-------
This module takes the raw structured output from vision.py (dense captions +
OCR text) and asks GPT to write a single, fluent, human-readable description
of the image — the kind a screen-reader user would actually want to hear.

Example input to GPT:
  "Dense captions: [a person sitting at a desk, a laptop with code on screen,
  a coffee mug]. OCR: 'def main():'"

Example output from GPT:
  "A developer sits at a desk working on code. Their laptop displays a Python
  function, and a coffee mug sits nearby."

WHY NOT JUST USE THE VISION CAPTIONS DIRECTLY?
-----------------------------------------------
Azure AI Vision captions are accurate but often mechanical — a list of detected
regions. GPT synthesizes these into a flowing description that gives context,
infers relationships between objects, and sounds natural when read aloud.

MODEL
-----
Primary:    GPT-5.4-mini (Azure AI Foundry, East US 2) — multimodal, low-latency
Fallback 1: gpt-4o-mini  (if gpt-5.4-mini quota is unavailable)
Fallback 2: gpt-4o       (last resort)

The deployment name is read from the AZURE_OPENAI_DEPLOYMENT_NAME env var.
Switching models is a one-line change in .env — never a code rewrite.

AZURE SDK
---------
Package: openai (v1.x) — the modern client supporting Azure endpoints via AzureOpenAI().
We do NOT use the deprecated openai.ChatCompletion.create() style.

PHASE 1 STATUS
--------------
Documented stub — Captioner class implemented in Phase 2.
"""

# Phase 2 implementation will go here.
#
# class Captioner:
#     def __init__(self):
#         # Initialize AzureOpenAI() client using:
#         #   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
#         # Read deployment name from AZURE_OPENAI_DEPLOYMENT_NAME.
#         ...
#
#     def describe(self, vision_output: dict) -> str:
#         # Build a prompt from the dense captions and OCR text.
#         # Call GPT with the prompt and a system message tuned for accessibility.
#         # Return the generated description string.
#         ...

"""
test_pipeline.py -- Unit tests for the Audible Frames pipeline.

HOW TO RUN
----------
  # Make sure your venv is active, then:
  pytest tests/test_pipeline.py -v

PHILOSOPHY: MOCKING
-------------------
These tests do NOT make real Azure API calls. Instead, they use "mocks" --
fake objects that pretend to be Azure clients but return pre-set values.

Why mock?
  - Tests run instantly (no network calls, no latency)
  - Tests work without real Azure credentials (CI/CD doesn't need your keys)
  - Tests are deterministic -- they always return the same values, so a
    failing test means YOUR code broke, not Azure's API being slow

How mocking works:
  unittest.mock.patch() temporarily replaces a real class or function with a
  MagicMock -- a fake object that lets you control what it returns.
  When the test is done, patch() restores the real thing automatically.

WHAT WE TEST
------------
  - VisionClient.analyze(): returns the right shape even with mock Azure response
  - Captioner.describe(): calls GPT correctly and returns a string
  - SpeechClient.synthesize(): returns bytes (not crashes)
  - run_pipeline(): stitches all three together correctly
  - /health endpoint: returns 200 OK
  - /describe endpoint: rejects bad inputs; returns audio for valid inputs
"""

import json
import pytest
from unittest.mock import patch, MagicMock

# We set fake env vars before importing our modules so no real credentials are needed.
import os
os.environ.setdefault("AZURE_VISION_ENDPOINT", "https://fake.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_VISION_KEY", "fake-vision-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "fake-openai-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ.setdefault("AZURE_SPEECH_KEY", "fake-speech-key")
os.environ.setdefault("AZURE_SPEECH_REGION", "eastus2")
os.environ.setdefault("AZURE_CONTENT_SAFETY_ENDPOINT", "https://fake.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_CONTENT_SAFETY_KEY", "fake-safety-key")


# ===========================================================================
# VisionClient tests
# ===========================================================================

class TestVisionClient:
    """Tests for src/vision.py -- the Azure AI Vision wrapper."""

    def test_analyze_returns_captions_and_ocr_keys(self):
        """
        The analyze() method must always return a dict with 'captions' and 'ocr_text'
        keys, even if Azure returns empty data.
        """
        with patch("src.vision.ImageAnalysisClient") as MockClient:
            # Set up a fake Vision result with one caption and no OCR text.
            mock_result = MagicMock()
            mock_cap = MagicMock()
            mock_cap.text = "a cat sitting on a sofa"
            mock_cap.confidence = 0.95
            mock_result.dense_captions.list = [mock_cap]
            mock_result.read = None  # no text in this image

            # Make the mock client's analyze() return our fake result.
            MockClient.return_value.analyze.return_value = mock_result

            from src.vision import VisionClient
            client = VisionClient()
            result = client.analyze(b"fake_image_bytes")

        # The result must have both keys regardless of Azure's response.
        assert "captions" in result, "Result must contain 'captions' key"
        assert "ocr_text" in result, "Result must contain 'ocr_text' key"

    def test_analyze_returns_caption_text(self):
        """Captions from Azure should appear in the returned captions list."""
        with patch("src.vision.ImageAnalysisClient") as MockClient:
            mock_result = MagicMock()
            mock_cap = MagicMock()
            mock_cap.text = "a developer working on a laptop"
            mock_cap.confidence = 0.92
            mock_result.dense_captions.list = [mock_cap]
            mock_result.read = None
            MockClient.return_value.analyze.return_value = mock_result

            from src.vision import VisionClient
            result = VisionClient().analyze(b"fake_bytes")

        assert "a developer working on a laptop" in result["captions"]

    def test_analyze_filters_low_confidence_captions(self):
        """
        Captions with confidence below 0.5 should be filtered out.
        Low-confidence captions add noise to the GPT prompt.
        """
        with patch("src.vision.ImageAnalysisClient") as MockClient:
            mock_result = MagicMock()
            high_conf = MagicMock()
            high_conf.text = "a dog in a park"
            high_conf.confidence = 0.85  # above threshold -- keep
            low_conf = MagicMock()
            low_conf.text = "possibly a bird or something"
            low_conf.confidence = 0.3   # below threshold -- discard
            mock_result.dense_captions.list = [high_conf, low_conf]
            mock_result.read = None
            MockClient.return_value.analyze.return_value = mock_result

            from src.vision import VisionClient
            result = VisionClient().analyze(b"fake_bytes")

        assert "a dog in a park" in result["captions"]
        assert "possibly a bird or something" not in result["captions"]

    def test_analyze_extracts_ocr_text(self):
        """Text found in the image via OCR should appear in ocr_text."""
        with patch("src.vision.ImageAnalysisClient") as MockClient:
            mock_result = MagicMock()
            mock_result.dense_captions = None

            # Mock the OCR block structure: blocks -> lines -> text
            mock_line = MagicMock()
            mock_line.text = "OPEN DAILY"
            mock_block = MagicMock()
            mock_block.lines = [mock_line]
            mock_result.read.blocks = [mock_block]
            MockClient.return_value.analyze.return_value = mock_result

            from src.vision import VisionClient
            result = VisionClient().analyze(b"fake_bytes")

        assert "OPEN DAILY" in result["ocr_text"]


# ===========================================================================
# Captioner tests
# ===========================================================================

class TestCaptioner:
    """Tests for src/captioner.py -- the GPT description generator."""

    def test_describe_returns_string(self):
        """describe() must return a non-empty string for any valid vision output."""
        with patch("src.captioner.AzureOpenAI") as MockOpenAI:
            # Set up a fake GPT response.
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "A cat rests on a comfortable sofa."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            from src.captioner import Captioner
            captioner = Captioner()
            result = captioner.describe({
                "captions": ["a cat on a sofa"],
                "ocr_text": ""
            })

        assert isinstance(result, str), "describe() must return a string"
        assert len(result) > 0, "describe() must return a non-empty string"
        assert result == "A cat rests on a comfortable sofa."

    def test_describe_handles_empty_vision_output(self):
        """
        describe() must not crash if vision returned nothing.
        (e.g. if the image was too dark or ambiguous for Vision to detect anything)
        """
        with patch("src.captioner.AzureOpenAI") as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "No visual content could be detected."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            from src.captioner import Captioner
            # This should NOT raise an exception even with empty inputs.
            result = Captioner().describe({"captions": [], "ocr_text": ""})

        assert isinstance(result, str)

    def test_describe_calls_gpt_with_correct_model(self):
        """GPT must be called with the deployment name from the env var."""
        with patch("src.captioner.AzureOpenAI") as MockOpenAI:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "A scene."
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response

            from src.captioner import Captioner
            Captioner().describe({"captions": ["something"], "ocr_text": ""})

            # Check that the model argument passed to GPT matches our env var.
            call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-5.4-mini"


# ===========================================================================
# SpeechClient tests
# ===========================================================================

class TestSpeechClient:
    """Tests for src/speech.py -- the Azure TTS wrapper."""

    def test_synthesize_returns_bytes(self):
        """synthesize() must return bytes (the WAV audio data)."""
        with patch("src.speech.speechsdk.SpeechConfig"), \
             patch("src.speech.speechsdk.audio.AudioOutputConfig"), \
             patch("src.speech.speechsdk.SpeechSynthesizer") as MockSynth, \
             patch("src.speech.pathlib.Path") as MockPath:

            # Fake a successful synthesis result.
            import azure.cognitiveservices.speech as speechsdk
            mock_result = MagicMock()
            mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
            MockSynth.return_value.speak_text_async.return_value.get.return_value = mock_result

            # Fake the temp file read -- return some bytes.
            mock_path_instance = MagicMock()
            mock_path_instance.read_bytes.return_value = b"fake_audio_data"
            mock_path_instance.exists.return_value = True
            MockPath.return_value = mock_path_instance

            from src.speech import SpeechClient
            audio = SpeechClient().synthesize("Hello, this is a test.")

        assert isinstance(audio, bytes), "synthesize() must return bytes"


# ===========================================================================
# Pipeline integration test (all mocked)
# ===========================================================================

class TestPipeline:
    """Integration test for src/pipeline.py -- tests that the three steps connect."""

    def test_run_pipeline_returns_expected_keys(self):
        """run_pipeline() must return a dict with description, audio_bytes, and timing."""
        with patch("src.pipeline.VisionClient") as MockVision, \
             patch("src.pipeline.Captioner") as MockCaptioner, \
             patch("src.pipeline.SpeechClient") as MockSpeech:

            MockVision.return_value.analyze.return_value = {
                "captions": ["a test image"],
                "ocr_text": ""
            }
            MockCaptioner.return_value.describe.return_value = "A test image is shown."
            MockSpeech.return_value.synthesize.return_value = b"fake_audio"

            from src.pipeline import run_pipeline
            result = run_pipeline(b"fake_image")

        assert "description" in result
        assert "audio_bytes" in result
        assert "timing" in result
        assert result["description"] == "A test image is shown."
        assert result["audio_bytes"] == b"fake_audio"

    def test_run_pipeline_timing_has_all_keys(self):
        """Timing dict must include per-step and total milliseconds."""
        with patch("src.pipeline.VisionClient") as MockVision, \
             patch("src.pipeline.Captioner") as MockCaptioner, \
             patch("src.pipeline.SpeechClient") as MockSpeech:

            MockVision.return_value.analyze.return_value = {"captions": [], "ocr_text": ""}
            MockCaptioner.return_value.describe.return_value = "A scene."
            MockSpeech.return_value.synthesize.return_value = b"audio"

            from src.pipeline import run_pipeline
            result = run_pipeline(b"img")

        timing = result["timing"]
        assert "vision_ms" in timing
        assert "captioner_ms" in timing
        assert "speech_ms" in timing
        assert "total_ms" in timing


# ===========================================================================
# API endpoint tests
# ===========================================================================

class TestAPI:
    """Tests for the FastAPI endpoints in src/api.py."""

    @pytest.fixture
    def client(self):
        """
        Create a test client for the FastAPI app.

        TestClient lets us make HTTP requests to the app without running a server.
        It's provided by the httpx library (which we installed in requirements.txt).
        """
        from fastapi.testclient import TestClient
        from src.api import app
        return TestClient(app)

    def test_health_returns_ok(self, client):
        """GET /health must return 200 with {"status": "ok"}."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_describe_rejects_non_image(self, client):
        """POST /describe must reject files that aren't images (415 Unsupported Media Type)."""
        # Upload a text file pretending to be an image -- should be rejected.
        response = client.post(
            "/describe",
            files={"file": ("document.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    def test_describe_rejects_empty_file(self, client):
        """POST /describe must reject empty files (400 Bad Request)."""
        response = client.post(
            "/describe",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_describe_returns_audio_for_valid_image(self, client):
        """
        POST /describe with a valid image and mocked pipeline should return
        a WAV audio response with the description in a header.
        """
        with patch("src.api.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "description": "A beautiful sunset over the ocean.",
                "audio_bytes": b"fake_wav_audio_bytes",
                "timing": {
                    "vision_ms": 300,
                    "captioner_ms": 900,
                    "speech_ms": 400,
                    "total_ms": 1600,
                }
            }

            # A minimal valid JPEG header (1x1 white pixel, not a real image
            # but enough to pass content_type validation).
            fake_jpeg = b"\xff\xd8\xff\xe0"  # JPEG magic bytes

            response = client.post(
                "/describe",
                files={"file": ("photo.jpg", fake_jpeg, "image/jpeg")},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Description" in response.headers
        assert response.headers["X-Description"] == "A beautiful sunset over the ocean."
        assert response.content == b"fake_wav_audio_bytes"

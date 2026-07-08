"""
speech.py -- Azure AI Speech (Text-to-Speech) integration.

PURPOSE
-------
Takes a text description and converts it to audio using Azure's neural TTS.
Returns raw WAV bytes that the API can stream directly to the browser.

WINDOWS FILE LOCK FIX
----------------------
The original version wrote audio to a temp file then read+deleted it.
On Windows, the Speech SDK keeps a file handle open after synthesis completes,
so deleting immediately causes [WinError 32] "file in use by another process."

The fix: use audio_config=None so the SDK returns audio bytes directly in
result.audio_data -- no temp file, no file locking, no cleanup needed.

OUTPUT FORMAT
-------------
Default: Riff16Khz16BitMonoPcm -- a standard WAV file (with header).
Universally supported by browsers, media players, and screen readers.
The <audio> element in the UI plays this as audio/wav.

AZURE SDK
---------
Package: azure-cognitiveservices-speech
"""

import os
import azure.cognitiveservices.speech as speechsdk


class SpeechClient:
    """
    Wraps Azure AI Speech to convert text to WAV audio bytes.

    Usage:
        client = SpeechClient()
        audio_bytes = client.synthesize("A developer sits at a desk.")
        # audio_bytes is a valid WAV file in bytes -- send it over HTTP or play it
    """

    def __init__(self):
        """
        Initialize the Speech client from environment variables.

        AZURE_SPEECH_KEY    -- API key for your Azure Speech resource
        AZURE_SPEECH_REGION -- region string, e.g. "eastus2"
                               (the SDK constructs the endpoint from this internally)
        """
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.environ["AZURE_SPEECH_KEY"],
            region=os.environ["AZURE_SPEECH_REGION"],
        )

        # Choose the TTS voice.
        # AriaNeural is a natural-sounding English voice, well-rated for accessibility.
        voice = os.environ.get("AZURE_SPEECH_VOICE", "en-US-AriaNeural")
        self.speech_config.speech_synthesis_voice_name = voice

        # Output format: standard WAV with header (16kHz, 16-bit mono PCM).
        # This is the default when audio_config=None, stated here explicitly for clarity.
        # The browser's <audio> element plays this as audio/wav without any conversion.
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech and return WAV audio as bytes.

        We pass audio_config=None so the SDK collects audio data in memory
        and returns it via result.audio_data. This avoids writing a temp file,
        which on Windows causes a file-locking error when we try to delete it.

        Args:
            text: The description to speak. Typically 2-4 sentences.

        Returns:
            Valid WAV audio bytes (includes WAV file header).
            Can be sent as-is with Content-Type: audio/wav.

        Raises:
            RuntimeError: If Azure cancels or fails the synthesis.
        """
        # audio_config=None tells the SDK: don't write to a file or audio device.
        # Instead, accumulate all audio in result.audio_data (bytes).
        # This completely avoids file I/O and the Windows file-locking problem.
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None,  # <-- key: in-memory, no temp file
        )

        # speak_text_async() sends the text to Azure.
        # .get() blocks until the entire audio has been received.
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # result.audio_data contains the complete WAV file as bytes.
            # No files to clean up -- everything stayed in memory.
            return bytes(result.audio_data)

        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(
                f"Speech synthesis cancelled. "
                f"Reason: {details.reason}. "
                f"Error: {details.error_details}. "
                f"Check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env."
            )
        else:
            raise RuntimeError(
                f"Speech synthesis failed. Reason code: {result.reason}"
            )

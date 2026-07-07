"""
speech.py -- Azure AI Speech (Text-to-Speech) integration.

PURPOSE
-------
Takes a text description (from captioner.py) and converts it to audio using
Azure's neural TTS voices. Returns raw audio bytes (WAV format) that the API
can send directly to the client.

WHY NEURAL TTS?
---------------
Older TTS systems sound robotic and are tiring to listen to. Azure's neural
voices are trained on real human speech and sound natural -- important for an
assistive tech app where users may listen to many descriptions per day.

Voice: en-US-AriaNeural -- a clear, natural English voice well-suited for
accessibility. Can be changed via env var for future internationalization.

OUTPUT FORMAT
-------------
WAV (PCM audio) -- lossless, universally supported by browsers, media players,
and screen readers. No encoding step needed, so it's simpler and faster than MP3.

CLEANUP
-------
This module writes audio to a temporary file on disk (the Speech SDK requires
a file path or stream to write to). The temp file is always deleted in a
`finally` block -- even if synthesis fails -- so no run leaves audio files
behind that accumulate on disk.

AZURE SDK
---------
Package: azure-cognitiveservices-speech
Windows: pip installs the correct Windows-specific wheel automatically.
"""

import os
import pathlib
import tempfile

# The Azure Speech SDK is imported as `speechsdk` (conventional alias).
import azure.cognitiveservices.speech as speechsdk


class SpeechClient:
    """
    Wraps Azure AI Speech to convert text to audio.

    Usage:
        client = SpeechClient()
        audio_bytes = client.synthesize("Hello, this is a description.")
        # audio_bytes is raw WAV audio you can write to a file or send over HTTP
    """

    def __init__(self):
        """
        Initialize the Speech client with credentials from environment variables.

        AZURE_SPEECH_KEY    -- your Azure Speech resource key
        AZURE_SPEECH_REGION -- the region your Speech resource is in (e.g. "eastus2")
                               Note: the Speech SDK uses the region string, NOT an endpoint URL.
        """
        speech_key = os.environ["AZURE_SPEECH_KEY"]
        speech_region = os.environ["AZURE_SPEECH_REGION"]

        # SpeechConfig holds all the settings for the synthesizer.
        # We pass the key and region -- the SDK constructs the endpoint internally.
        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )

        # Choose the TTS voice. AriaNeural is natural-sounding and widely used
        # for accessibility applications.
        # Full list of voices: https://learn.microsoft.com/azure/ai-services/speech-service/language-support
        voice = os.environ.get("AZURE_SPEECH_VOICE", "en-US-AriaNeural")
        self.speech_config.speech_synthesis_voice_name = voice

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech and return the audio as bytes.

        The SDK writes audio to a temporary WAV file, then we read those bytes
        and delete the file. This is the most reliable cross-platform approach.

        Args:
            text: The description to speak aloud. Should be 1-5 sentences.

        Returns:
            Raw WAV audio bytes. Can be sent directly as an HTTP response
            with Content-Type: audio/wav.

        Raises:
            RuntimeError: If Azure Speech synthesis fails or is cancelled.
        """
        # Create a temporary file path with a .wav extension.
        # tempfile.mktemp() gives us a unique path -- it doesn't create the file yet,
        # just reserves a name. The SDK will create the file when it writes audio.
        tmp_path = pathlib.Path(tempfile.mktemp(suffix=".wav"))

        try:
            # AudioOutputConfig tells the SDK WHERE to write the audio.
            # filename= means: write audio to this file on disk.
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(tmp_path))

            # SpeechSynthesizer is the main TTS object.
            # It needs both the config (voice, format) and where to write output.
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config,
            )

            # speak_text_async() sends the text to Azure and starts synthesis.
            # .get() blocks until synthesis is complete (or fails).
            result = synthesizer.speak_text_async(text).get()

            # Check the result reason to know if it worked.
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Success -- read the audio file and return its bytes.
                return tmp_path.read_bytes()

            elif result.reason == speechsdk.ResultReason.Canceled:
                # Azure cancelled the request -- usually a credentials or quota issue.
                # cancellation_details has the specific reason and error message.
                details = result.cancellation_details
                raise RuntimeError(
                    f"Speech synthesis was cancelled. "
                    f"Reason: {details.reason}. "
                    f"Error details: {details.error_details}. "
                    f"Check your AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env."
                )
            else:
                raise RuntimeError(
                    f"Speech synthesis failed with unexpected reason: {result.reason}"
                )

        finally:
            # ALWAYS clean up the temp file, even if synthesis raised an exception.
            # This is what `finally` is for -- it runs no matter what.
            # Without this, every failed run would leave a .wav file on disk.
            if tmp_path.exists():
                tmp_path.unlink()
                # .unlink() is pathlib's way of deleting a file (equivalent to os.remove())

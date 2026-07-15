"""
telemetry.py -- Azure Application Insights observability.

PURPOSE
-------
Once deployed, we need to know:
  - How many requests are coming in?
  - How long does each step take? (Vision vs GPT vs Speech)
  - Which requests are failing, and why?
  - Is cost growing faster than expected?

Application Insights collects all of this automatically once we configure it.
You get a live dashboard in the Azure portal showing charts of latency,
error rates, and request volume -- no manual log-digging needed.

HOW IT WORKS
------------
We use Python's built-in `logging` module, but attach an Azure handler to it.
Every time we call logger.info() or logger.error() in api.py, the log line
goes to both:
  1. Your terminal (for local development)
  2. Azure Application Insights (when the env var is set)

We also track custom "events" with structured data (like timing breakdowns)
so they're queryable in the Azure portal with Kusto Query Language (KQL).

ENVIRONMENT VARIABLES
---------------------
  APPLICATIONINSIGHTS_CONNECTION_STRING  -- from Azure portal (Application Insights resource)
  If not set, telemetry is silently skipped (works fine for local dev).

PHASE 5 SETUP
-------------
You still need to:
  1. Create an Application Insights resource in Azure (free tier is fine)
  2. Copy the Connection String from the resource's Overview page
  3. Add it to your .env file as APPLICATIONINSIGHTS_CONNECTION_STRING
"""

import os
import logging
import json


logger = logging.getLogger(__name__)


def setup_telemetry() -> bool:
    """
    Configure Azure Application Insights logging if the connection string is set.

    Call this once at application startup (in api.py).

    Returns:
        True if Application Insights was configured, False if running locally
        without the connection string (safe to ignore).
    """
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()

    if not connection_string:
        # No connection string = local development mode.
        # Everything works fine -- logs just go to the terminal.
        logger.debug(
            "APPLICATIONINSIGHTS_CONNECTION_STRING not set -- "
            "telemetry disabled (fine for local dev)."
        )
        return False

    try:
        # azure-monitor-opentelemetry-exporter sends Python logs to App Insights.
        # It hooks into the root logger so all existing logger.info() calls
        # are automatically sent -- no code changes needed in other files.
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = AzureMonitorTraceExporter(connection_string=connection_string)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        logger.info("Application Insights telemetry configured.")
        return True

    except ImportError:
        # Package not installed -- silently skip.
        # This happens if someone runs locally without installing the extra package.
        logger.warning(
            "azure-monitor-opentelemetry-exporter not installed. "
            "Telemetry disabled. Run: pip install azure-monitor-opentelemetry-exporter"
        )
        return False

    except Exception as exc:
        # Don't crash the app if telemetry setup fails -- it's not critical.
        logger.warning(f"Failed to set up Application Insights: {exc}")
        return False


def track_pipeline_request(
    filename: str,
    file_size_bytes: int,
    timing: dict,
    description_length: int,
    success: bool,
    error: str = None,
):
    """
    Log a structured pipeline event to Application Insights.

    This creates a single log entry with all the relevant data for one request.
    In the Azure portal, you can query these with:

      traces
      | where message contains "pipeline_request"
      | project timestamp, customDimensions
      | order by timestamp desc

    Args:
        filename:           Original uploaded filename (e.g. "photo.jpg")
        file_size_bytes:    Size of uploaded image in bytes
        timing:             Dict from pipeline.run_pipeline() with latency per step
        description_length: Number of characters in the generated description
        success:            True if pipeline completed without error
        error:              Error message if success=False
    """
    event = {
        "event":              "pipeline_request",
        "filename":           filename,
        "file_size_kb":       round(file_size_bytes / 1024, 1),
        "success":            success,
        "total_ms":           timing.get("total_ms", 0),
        "vision_ms":          timing.get("vision_ms", 0),
        "captioner_ms":       timing.get("captioner_ms", 0),
        "speech_ms":          timing.get("speech_ms", 0),
        "description_chars":  description_length,
    }

    if error:
        event["error"] = error

    # Log as structured JSON so it's queryable in App Insights
    if success:
        logger.info(f"pipeline_request {json.dumps(event)}")
    else:
        logger.error(f"pipeline_request {json.dumps(event)}")


def track_safety_rejection(filename: str, categories: dict):
    """
    Log when an image is rejected by Content Safety.

    Useful for spotting patterns -- if many rejections come from one user
    or one type of content, this shows up in the App Insights dashboard.
    """
    event = {
        "event":      "safety_rejection",
        "filename":   filename,
        "categories": categories,
    }
    logger.warning(f"safety_rejection {json.dumps(event)}")

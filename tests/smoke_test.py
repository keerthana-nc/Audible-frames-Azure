"""
smoke_test.py — Import verification for the full dependency stack.

PURPOSE
-------
Run this script immediately after `pip install -r requirements.txt` and
BEFORE writing any feature code. If any import fails, the dependency stack
is broken and needs to be fixed first — otherwise you'll get mysterious
errors later that are hard to trace back to a missing package.

HOW TO RUN
----------
  # Make sure your virtual environment is active first:
  #   Windows: venv\\Scripts\\activate
  #
  python tests/smoke_test.py

WHAT COUNTS AS PASSING
-----------------------
Every line shows "✓" and the script exits with:
  "All imports OK — stack is healthy, ready for Phase 2."

If ANY line shows "✗", fix that package before proceeding.

COMMON FIXES
------------
  ✗ azure.cognitiveservices.speech
    → pip install azure-cognitiveservices-speech --upgrade
    → On Windows, make sure you're using Python 3.11 (not 3.12+)
      because the Speech SDK wheel may not be available for newer versions yet.

  ✗ azure.ai.vision.imageanalysis
    → pip install azure-ai-vision-imageanalysis --upgrade

  ✗ Any pydantic error
    → pip install "pydantic>=2.0" --upgrade
    → Never install pydantic v1 — it breaks FastAPI 0.100+
"""

import sys


def check(module_name: str, display_name: str = "") -> bool:
    """
    Try to import a module by name and print the result.

    Args:
        module_name:  The Python import path (e.g. "azure.ai.vision.imageanalysis")
        display_name: A friendlier name to show in the output (e.g. "azure-ai-vision-imageanalysis")

    Returns:
        True if the import succeeded, False if it failed.
    """
    label = display_name or module_name
    try:
        mod = __import__(module_name)
        # Walk dotted paths: __import__("azure.ai.vision") returns the top-level
        # "azure" module; we need to traverse .ai.vision to get the real module.
        for part in module_name.split(".")[1:]:
            mod = getattr(mod, part)
        version = getattr(mod, "__version__", "version not exposed")
        print(f"  ✓ {label:<45} ({version})")
        return True
    except ImportError as exc:
        print(f"  ✗ {label:<45} FAILED: {exc}")
        return False


def main():
    print(f"\nPython {sys.version}")
    print(f"Executable: {sys.executable}\n")
    print("Checking all imports...\n")

    results = []

    # ── Web framework ──────────────────────────────────────────────────────────
    results.append(check("fastapi", "fastapi"))
    results.append(check("uvicorn", "uvicorn"))
    results.append(check("pydantic", "pydantic (must be v2.x)"))
    results.append(check("multipart", "python-multipart"))
    results.append(check("httpx", "httpx"))
    results.append(check("dotenv", "python-dotenv"))

    print()

    # ── Image handling ─────────────────────────────────────────────────────────
    results.append(check("PIL", "pillow"))

    print()

    # ── Azure SDKs ─────────────────────────────────────────────────────────────
    results.append(check("azure.ai.vision.imageanalysis", "azure-ai-vision-imageanalysis"))
    results.append(check("azure.identity", "azure-identity"))
    results.append(check("azure.ai.contentsafety", "azure-ai-contentsafety"))
    results.append(check("azure.cognitiveservices.speech", "azure-cognitiveservices-speech"))

    print()

    # ── OpenAI ────────────────────────────────────────────────────────────────
    results.append(check("openai", "openai (must be v1.x)"))

    print()

    # ── Testing ───────────────────────────────────────────────────────────────
    results.append(check("pytest", "pytest"))
    results.append(check("pytest_asyncio", "pytest-asyncio"))

    print()

    # ── Evaluation (Phase 3) ──────────────────────────────────────────────────
    results.append(check("nltk", "nltk"))
    results.append(check("rouge_score", "rouge-score"))

    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    failures = [r for r in results if not r]
    if not failures:
        print("All imports OK — stack is healthy, ready for Phase 2.")
        sys.exit(0)
    else:
        print(f"{len(failures)} import(s) FAILED. Fix these before proceeding to Phase 2.")
        print("See the COMMON FIXES section at the top of this file for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()

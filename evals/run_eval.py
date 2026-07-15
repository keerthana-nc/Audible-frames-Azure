"""
run_eval.py -- Evaluation harness for Audible Frames.

PURPOSE
-------
Measures whether our Azure pipeline produces better image descriptions than a
single HuggingFace baseline model. This is what backs up the resume claim:
  "improving caption quality by X% on METEOR and ROUGE-L over a single HuggingFace model"

The number this script prints is the real number. Nothing is faked or targeted.

HOW IT WORKS
------------
For each image in the dataset (30 COCO 2017 val images):

  1. BASELINE: Run nlpconnect/vit-gpt2-image-captioning (HuggingFace, free, local)
     -- a standard academic baseline for image captioning

  2. AZURE PIPELINE: Run Azure AI Vision -> GPT-5.4-mini (our pipeline)
     -- we skip Speech here since METEOR/ROUGE measure text, not audio

  3. METRICS (automatic):
     -- METEOR: measures semantic similarity using word alignment + synonyms
     -- ROUGE-L: measures longest common subsequence overlap
     Both compare generated caption against the 5 human reference captions.
     Higher = better. Standard captioning evaluation methodology.

  4. LLM-AS-JUDGE (GPT scores both captions):
     -- Rates each caption 1-5 on accuracy, completeness, and usefulness for blind users
     -- This is the "gold standard" eval signal for 2026 hiring screens

OUTPUT
------
  evals/results/results_<timestamp>.json  -- full per-image data
  Console: summary table with average scores

HOW TO RUN
----------
  # Install eval packages first (one time):
  pip install -r evals/requirements_eval.txt

  # Download dataset first (one time):
  python evals/prepare_dataset.py

  # Run the eval (takes ~10-15 minutes, makes real Azure API calls):
  python evals/run_eval.py

  # Run without Azure calls (checks baseline only, much faster):
  python evals/run_eval.py --baseline-only

COST ESTIMATE
-------------
  30 images x Vision (F0 free) + GPT-5.4-mini (~100 tokens in + ~100 out) + LLM judge (~300 tokens)
  Total: ~12,000 GPT tokens = roughly $0.01-0.05 depending on your plan.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import datetime

from dotenv import load_dotenv
load_dotenv()

# Add project root to path so 'from src.vision import ...' works when running
# this script from any directory (not just the project root).
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Paths
DATASET_DIR   = pathlib.Path(__file__).parent / "dataset"
CAPTIONS_FILE = DATASET_DIR / "captions.json"
RESULTS_DIR   = pathlib.Path(__file__).parent / "results"


# =============================================================================
# Step 1 -- HuggingFace baseline
# =============================================================================

def load_baseline_model():
    """
    Load the VIT-GPT2 image captioning model from HuggingFace.

    Why this model?
      - nlpconnect/vit-gpt2-image-captioning is a standard, lightweight baseline
      - Vision Transformer (ViT) encoder + GPT-2 decoder
      - Runs locally -- no API calls, no cost
      - Widely used in captioning papers as a comparison point

    It's intentionally simple. Our Azure pipeline (Vision + GPT-5.4-mini) should
    produce richer, more contextual captions. The eval shows by how much.
    """
    try:
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        import torch
    except ImportError:
        print("ERROR: transformers/torch not installed.")
        print("Run: pip install -r evals/requirements_eval.txt")
        sys.exit(1)

    print("Loading HuggingFace baseline model (vit-gpt2-image-captioning)...")
    print("(First run downloads ~900MB -- cached after that)\n")

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model     = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, tokenizer, device


def run_baseline(model, processor, tokenizer, device, image_path: str) -> str:
    """
    Generate a caption for an image using the HuggingFace baseline model.

    Returns:
        A single caption string.
    """
    from PIL import Image
    import torch

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=[img], return_tensors="pt").pixel_values.to(device)

    # Generate caption -- max 64 tokens is enough for a one-sentence description
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=64, num_beams=1)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# =============================================================================
# Step 2 -- Azure pipeline (Vision + GPT only, skip Speech for eval)
# =============================================================================

def run_azure_caption(image_path: str) -> str:
    """
    Run Vision + GPT on an image and return the text description.

    We skip the Speech step here because METEOR/ROUGE measure text quality,
    not audio quality. Skipping Speech also saves time and cost.
    """
    from src.vision import VisionClient
    from src.captioner import Captioner

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    vision_output = VisionClient().analyze(image_bytes)
    description   = Captioner().describe(vision_output)
    return description


# =============================================================================
# Step 3 -- Automatic metrics (METEOR + ROUGE-L)
# =============================================================================

def compute_meteor(hypothesis: str, references: list[str]) -> float:
    """
    Compute METEOR score for a generated caption against reference captions.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering):
      - Considers exact matches, stemmed matches, and synonyms (via WordNet)
      - More robust than BLEU for short texts
      - Higher is better. Range: 0.0 to 1.0

    We take the max METEOR against all 5 references (standard practice).
    """
    import nltk
    # Download required NLTK data if not already present
    for resource in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    hyp_tokens = word_tokenize(hypothesis.lower())
    scores = []
    for ref in references:
        ref_tokens = word_tokenize(ref.lower())
        scores.append(meteor_score([ref_tokens], hyp_tokens))

    return max(scores) if scores else 0.0


def compute_rouge_l(hypothesis: str, references: list[str]) -> float:
    """
    Compute ROUGE-L F1 score for a generated caption against reference captions.

    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence):
      - Measures the longest common subsequence between hypothesis and reference
      - Captures in-order word matches without requiring consecutive n-grams
      - Higher is better. Range: 0.0 to 1.0

    We take the max ROUGE-L against all 5 references.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hypothesis)["rougeL"].fmeasure for ref in references]
    return max(scores) if scores else 0.0


# =============================================================================
# Step 4 -- LLM-as-judge
# =============================================================================

def llm_judge(image_description: str, model_label: str) -> dict:
    """
    Ask GPT to grade a caption on three dimensions relevant to accessibility.

    The LLM-as-judge approach is the strongest 2026 eval signal for hiring screens
    because it captures qualities that METEOR/ROUGE miss:
      - Does the description convey the scene accurately?
      - Does it mention the most important elements?
      - Would a blind user find this genuinely useful?

    Returns:
        {
            "accuracy": 4,        # 1-5: does it match what's in the image?
            "completeness": 3,    # 1-5: does it cover the main elements?
            "usefulness": 4,      # 1-5: helpful for a blind/low-vision user?
            "average": 3.67,
            "reasoning": "..."    # GPT's explanation
        }
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    prompt = f"""You are evaluating image descriptions for an accessibility application that helps blind and low-vision users understand images.

Rate the following description on three dimensions, each from 1 to 5:
  1. Accuracy (1=wrong/misleading, 5=perfectly accurate)
  2. Completeness (1=misses most elements, 5=covers all key elements)
  3. Usefulness for blind users (1=not helpful, 5=very helpful, gives real understanding)

Description to evaluate ({model_label}):
"{image_description}"

Respond in this exact JSON format (no other text):
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "usefulness": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}}"""

    try:
        response = client.responses.create(
            model=deployment,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=200,
            temperature=0.2,   # low temperature for consistent scoring
        )
        raw = response.output_text.strip()

        # Strip markdown code fences if present (e.g. ```json ... ```)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)
        scores["average"] = round(
            (scores["accuracy"] + scores["completeness"] + scores["usefulness"]) / 3, 2
        )
        return scores

    except Exception as e:
        # If judge fails, return neutral scores rather than crashing the whole eval
        return {"accuracy": 0, "completeness": 0, "usefulness": 0, "average": 0.0,
                "reasoning": f"Judge failed: {e}"}


# =============================================================================
# Main eval loop
# =============================================================================

def run_eval(baseline_only: bool = False):
    """
    Run the full evaluation loop and save results to JSON.

    Args:
        baseline_only: If True, skip Azure API calls (faster, no cost, for testing).
    """
    # --- Load dataset ---
    if not CAPTIONS_FILE.exists():
        print("ERROR: Dataset not found. Run this first:")
        print("  python evals/prepare_dataset.py")
        sys.exit(1)

    records = json.loads(CAPTIONS_FILE.read_text())
    print(f"Loaded {len(records)} images from dataset.\n")

    if not baseline_only:
        # Load baseline model once (reuse across all images)
        model, processor, tokenizer, device = load_baseline_model()

    results = []
    baseline_meteors, baseline_rouges = [], []
    azure_meteors,    azure_rouges    = [], []

    for i, record in enumerate(records):
        image_path = record["image_path"]
        refs       = record["reference_captions"]

        print(f"[{i+1:2d}/{len(records)}] {pathlib.Path(image_path).name}")

        result = {
            "id":                  record["id"],
            "image_path":          image_path,
            "reference_captions":  refs,
        }

        # -- Baseline --
        print("  Running HuggingFace baseline...", end=" ", flush=True)
        t0 = time.monotonic()
        baseline_caption = run_baseline(model, processor, tokenizer, device, image_path)
        baseline_ms = round((time.monotonic() - t0) * 1000)
        print(f"done ({baseline_ms}ms)")
        print(f"  Baseline: {baseline_caption}")

        baseline_meteor = compute_meteor(baseline_caption, refs)
        baseline_rouge  = compute_rouge_l(baseline_caption, refs)
        baseline_meteors.append(baseline_meteor)
        baseline_rouges.append(baseline_rouge)

        result["baseline"] = {
            "caption":   baseline_caption,
            "meteor":    round(baseline_meteor, 4),
            "rouge_l":   round(baseline_rouge, 4),
            "latency_ms": baseline_ms,
        }

        if not baseline_only:
            # -- Azure pipeline --
            print("  Running Azure pipeline...", end=" ", flush=True)
            t0 = time.monotonic()
            try:
                azure_caption = run_azure_caption(image_path)
                azure_ms = round((time.monotonic() - t0) * 1000)
                print(f"done ({azure_ms}ms)")
                print(f"  Azure:    {azure_caption}")

                azure_meteor = compute_meteor(azure_caption, refs)
                azure_rouge  = compute_rouge_l(azure_caption, refs)
                azure_meteors.append(azure_meteor)
                azure_rouges.append(azure_rouge)

                # -- LLM-as-judge --
                print("  LLM judging both captions...", end=" ", flush=True)
                baseline_judge = llm_judge(baseline_caption, "HuggingFace vit-gpt2 baseline")
                azure_judge    = llm_judge(azure_caption,    "Azure Vision + GPT-5.4-mini")
                print("done")

                result["azure"] = {
                    "caption":    azure_caption,
                    "meteor":     round(azure_meteor, 4),
                    "rouge_l":    round(azure_rouge, 4),
                    "latency_ms": azure_ms,
                    "llm_judge":  azure_judge,
                }
                result["baseline"]["llm_judge"] = baseline_judge

            except Exception as e:
                print(f"FAILED: {e}")
                result["azure"] = {"error": str(e)}

        results.append(result)
        print()

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"

    # Compute summary statistics
    summary = {
        "dataset":       "COCO 2017 validation (30 images)",
        "citation":      "Lin et al., 2014. https://arxiv.org/abs/1405.0312",
        "baseline_model": "nlpconnect/vit-gpt2-image-captioning",
        "azure_model":   os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "unknown"),
        "num_images":    len(results),
        "baseline": {
            "avg_meteor": round(sum(baseline_meteors) / len(baseline_meteors), 4) if baseline_meteors else 0,
            "avg_rouge_l": round(sum(baseline_rouges) / len(baseline_rouges), 4) if baseline_rouges else 0,
        },
    }

    if azure_meteors:
        avg_baseline_meteor = sum(baseline_meteors) / len(baseline_meteors)
        avg_azure_meteor    = sum(azure_meteors)    / len(azure_meteors)
        avg_baseline_rouge  = sum(baseline_rouges)  / len(baseline_rouges)
        avg_azure_rouge     = sum(azure_rouges)     / len(azure_rouges)

        meteor_improvement = (avg_azure_meteor - avg_baseline_meteor) / avg_baseline_meteor * 100
        rouge_improvement  = (avg_azure_rouge  - avg_baseline_rouge)  / avg_baseline_rouge  * 100

        summary["azure"] = {
            "avg_meteor":  round(avg_azure_meteor, 4),
            "avg_rouge_l": round(avg_azure_rouge,  4),
        }
        summary["improvement"] = {
            "meteor_pct":  round(meteor_improvement, 1),
            "rouge_l_pct": round(rouge_improvement,  1),
        }

    output = {"summary": summary, "results": results}
    output_file.write_text(json.dumps(output, indent=2))

    # --- Print summary table ---
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Azure Pipeline':>14}")
    print("-" * 60)
    print(f"{'METEOR (avg)':<20} {summary['baseline']['avg_meteor']:>12.4f}", end="")
    if "azure" in summary:
        print(f" {summary['azure']['avg_meteor']:>14.4f}")
        print(f"{'ROUGE-L (avg)':<20} {summary['baseline']['avg_rouge_l']:>12.4f} {summary['azure']['avg_rouge_l']:>14.4f}")
        print()
        print(f"METEOR improvement:  {summary['improvement']['meteor_pct']:+.1f}%")
        print(f"ROUGE-L improvement: {summary['improvement']['rouge_l_pct']:+.1f}%")
    else:
        print()
        print(f"{'ROUGE-L (avg)':<20} {summary['baseline']['avg_rouge_l']:>12.4f}")
    print("=" * 60)
    print(f"\nFull results saved to: {output_file}")
    print("Copy the improvement numbers into your README results table.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Audible Frames evaluation.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip Azure API calls (tests baseline only, no cost)"
    )
    args = parser.parse_args()
    run_eval(baseline_only=args.baseline_only)

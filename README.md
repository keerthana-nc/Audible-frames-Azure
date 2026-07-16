# Audible Frames — Azure

> **Assistive tech:** converts any image into a natural-language audio description so people who are blind or have low vision can "hear" what's in a photo.

**Pipeline:**  
`Image` → Azure AI Vision *(dense captions + OCR)* → Azure OpenAI GPT-5.4-mini *(rich scene description)* → Azure AI Speech *(neural TTS)* → **Audio**

---

## Architecture

*Diagram coming in Phase 7.*

## Eval results

Evaluated on 30 MS COCO 2017 validation images with 5 human reference captions each.  
Baseline: `nlpconnect/vit-gpt2-image-captioning` (local HuggingFace model, no API cost).

| Metric   | HuggingFace baseline | Azure pipeline | Δ |
|----------|----------------------|----------------|---|
| METEOR   | 0.3613               | 0.3766         | **+4.2%** |
| ROUGE-L  | 0.4665               | 0.2460         | -47.3% |

**Why ROUGE-L is lower:** ROUGE-L measures the longest common subsequence between generated and reference captions. Human COCO references are short and generic (e.g. *"a woman walking down a street with a pigeon"*). The Azure pipeline produces long, detailed descriptions (e.g. *"A woman sits on a bench outdoors, facing a small group of pigeons on the ground in front of her. The scene is in black and white..."*). Longer, richer text scores lower on ROUGE-L even when it is objectively more useful — which is why METEOR (which accounts for synonyms and word alignment) and LLM-as-judge are better signals for accessibility-focused captioning.

**LLM-as-judge** (GPT-5.4-mini rating accuracy / completeness / usefulness 1–5): Azure pipeline consistently rated higher than baseline on all three dimensions across all 30 images. See `evals/results/` for full per-image breakdowns.

## Live demo

**https://audible-frames-app.salmonflower-7eea7334.eastus2.azurecontainerapps.io/docs**

Upload any image to the `/describe` endpoint and get back a natural-language audio description.

---

## Local setup

### Prerequisites
- Python 3.11
- An Azure account with the resources in `.env.example` provisioned

### Install
```bash
# 1. Clone the repo
git clone https://github.com/keerthana-nc/audible-frames-azure.git
cd audible-frames-azure

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the stack (run this before anything else)
python tests/smoke_test.py

# 5. Copy the env template and fill in your Azure keys
copy .env.example .env        # Windows
# cp .env.example .env        # Mac / Linux
```

### Run locally
```bash
uvicorn src.api:app --reload
# Open http://localhost:8000/docs to test the API
```

---

## Costs & cleanup

*Full cost breakdown coming in Phase 6. Short version:*

| Resource | Billing model |
|---|---|
| Azure AI Vision | Pay per 1,000 transactions (F0 free tier: 5,000/month) |
| Azure OpenAI (GPT-5.4-mini) | Pay per token used |
| Azure AI Speech | Pay per character synthesized (F0 free: 500K chars/month) |
| Azure AI Content Safety | Pay per 1,000 transactions (F0 free tier available) |
| Azure Container Registry | Billed for existing (~$5/month for Basic) |
| Azure Container Apps | Billed per request + CPU/memory when running (scale to 0 = $0 idle) |

### Stop charges without deleting
```bash
scripts\pause.bat
```

### Delete everything (irreversible)
```bash
scripts\teardown.bat
# WARNING: this deletes the entire resource group and ALL resources in it.
# You cannot undo this.
```

---

## Repository topics
`azure` · `gpt-4o` · `multimodal` · `text-to-speech` · `llm-evaluation` · `accessibility`

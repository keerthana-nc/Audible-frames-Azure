"""
prepare_dataset.py -- Download 30 COCO 2017 validation images + reference captions.

PURPOSE
-------
The evaluation harness needs a set of images with known "ground truth" captions
to measure how well our pipeline describes images compared to a baseline.

We use MS COCO 2017 validation set because:
  - It's the standard academic benchmark for image captioning
  - Each image has 5 human-written reference captions (more is better for metrics)
  - Publicly available, no login required
  - Widely cited in research papers

WHAT THIS SCRIPT DOES
---------------------
  1. Loads 30 COCO 2017 validation examples via HuggingFace datasets
  2. Saves each image as a JPEG in evals/dataset/images/
  3. Saves image paths + 5 reference captions to evals/dataset/captions.json

HOW TO RUN
----------
  # Make sure eval packages are installed first:
  pip install -r evals/requirements_eval.txt

  # Then run from the project root:
  python evals/prepare_dataset.py

CITATION
--------
  Microsoft COCO: Common Objects in Context
  Lin et al., 2014. https://arxiv.org/abs/1405.0312
  Dataset: https://cocodataset.org
"""

import json
import pathlib
import sys

# Number of images to download. 30 gives reliable metric estimates without
# spending too much on Azure API calls during eval.
NUM_IMAGES = 30

# Where to save everything
DATASET_DIR  = pathlib.Path(__file__).parent / "dataset"
IMAGES_DIR   = DATASET_DIR / "images"
CAPTIONS_FILE = DATASET_DIR / "captions.json"


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {NUM_IMAGES} COCO 2017 validation examples from HuggingFace...")
    print("(First run may take a minute to download -- cached after that)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install -r evals/requirements_eval.txt")
        sys.exit(1)

    # Load COCO captions from HuggingFace.
    # trust_remote_code is no longer supported in datasets>=5.0 -- omit it.
    # We try several known sources in order; fall back to direct COCO download.
    dataset = None
    sources = [
        ("jxie/coco_captions",        "validation"),
        ("phiyodr/coco2017",          "validation"),
        ("HuggingFaceM4/COCO",        "validation"),
    ]

    for dataset_name, split in sources:
        try:
            print(f"Trying {dataset_name}...")
            dataset = load_dataset(
                dataset_name,
                split=f"{split}[:{NUM_IMAGES}]",
            )
            print(f"Loaded from {dataset_name}\n")
            break
        except Exception as e:
            print(f"  Could not load {dataset_name}: {e}")
            continue

    if dataset is None:
        # Final fallback: download directly from COCO servers.
        # captions_val2017.json lists every image ID + 5 human captions.
        # We download 30 images by their real COCO filenames.
        print("\nHuggingFace sources unavailable -- falling back to direct COCO download...")
        _download_coco_direct()
        return


def _download_coco_direct():
    """
    Download 30 COCO val2017 images and their captions directly from
    cocodataset.org. Used as a fallback when HuggingFace sources are down.
    """
    import urllib.request
    import zipfile
    import io

    IMAGES_DIR    = DATASET_DIR / "images"
    CAPTIONS_FILE = DATASET_DIR / "captions.json"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Download captions annotation JSON (~241 MB zip; we stream and extract only
    # the captions file so we don't save the full zip to disk).
    ANNOTATIONS_URL = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    CAPTIONS_JSON   = "annotations/captions_val2017.json"
    IMAGE_BASE_URL  = "http://images.cocodataset.org/val2017/"

    print("Downloading COCO annotations zip (241 MB -- this may take a few minutes)...")
    with urllib.request.urlopen(ANNOTATIONS_URL) as resp:
        zip_bytes = io.BytesIO(resp.read())

    print("Extracting captions_val2017.json from zip...")
    with zipfile.ZipFile(zip_bytes) as zf:
        captions_data = json.loads(zf.read(CAPTIONS_JSON))

    # Build a dict: image_id -> list of caption strings
    from collections import defaultdict
    cap_map = defaultdict(list)
    for ann in captions_data["annotations"]:
        cap_map[ann["image_id"]].append(ann["caption"])

    # Take the first NUM_IMAGES images from the val set
    images_meta = captions_data["images"][:NUM_IMAGES]

    records = []
    for i, img_meta in enumerate(images_meta):
        image_id   = img_meta["id"]
        filename   = img_meta["file_name"]
        image_url  = IMAGE_BASE_URL + filename
        image_path = IMAGES_DIR / f"{i:04d}.jpg"

        print(f"  [{i+1:2d}/{NUM_IMAGES}] Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(image_url, image_path)
            refs = cap_map.get(image_id, ["no caption available"])[:5]
            records.append({
                "id":                 f"{i:04d}",
                "image_path":         str(image_path),
                "reference_captions": refs,
            })
            print(f"done ({len(refs)} captions)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    CAPTIONS_FILE.write_text(json.dumps(records, indent=2))
    print(f"\nDone. {len(records)} images saved to {IMAGES_DIR}")
    print(f"Captions index saved to {CAPTIONS_FILE}")
    print("\nNext step: python evals/run_eval.py")

    records = []
    for i, item in enumerate(dataset):
        image_id = f"{i:04d}"
        image_path = IMAGES_DIR / f"{image_id}.jpg"

        # Save the image.
        # item["image"] is a PIL Image object from the datasets library.
        img = item.get("image") or item.get("img")
        if img is None:
            print(f"  [{i+1}] WARNING: No image field found, skipping.")
            continue

        # Convert to RGB (some images may be RGBA or grayscale)
        img = img.convert("RGB")
        img.save(image_path, "JPEG", quality=90)

        # Extract reference captions.
        # Different datasets use different field names -- try all common ones.
        captions_raw = (
            item.get("captions")        # list of strings
            or item.get("sentences")    # some datasets use this
            or item.get("caption")      # singular form (wrap in list)
        )

        if isinstance(captions_raw, str):
            # Single caption string -- wrap in list
            reference_captions = [captions_raw]
        elif isinstance(captions_raw, list):
            if captions_raw and isinstance(captions_raw[0], dict):
                # List of dicts with "raw" key (COCO format)
                reference_captions = [c.get("raw", c.get("caption", "")) for c in captions_raw]
            else:
                # List of strings
                reference_captions = captions_raw
        else:
            reference_captions = ["no reference caption available"]

        records.append({
            "id": image_id,
            "image_path": str(image_path),
            "reference_captions": reference_captions,
        })

        print(f"  [{i+1:2d}/{NUM_IMAGES}] Saved {image_path.name} | {len(reference_captions)} reference captions")

    # Save the index file
    CAPTIONS_FILE.write_text(json.dumps(records, indent=2))

    print(f"\nDone. {len(records)} images saved to {IMAGES_DIR}")
    print(f"Captions index saved to {CAPTIONS_FILE}")
    print("\nNext step: python evals/run_eval.py")


if __name__ == "__main__":
    main()

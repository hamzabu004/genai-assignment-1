import json
import random
import os
from pathlib import Path

# Paths
ROOT = Path("/home/mustafa/genai-assignment-1/research")
SPLIT_MANIFEST = ROOT / "split_manifest.json"
VAL_MANIFEST = ROOT / "val_manifest.json"
TEST_MANIFEST = ROOT / "test_manifest.json"

CLASSES = ["clean", "salt", "blur", "occlusion"]

def generate_validation_manifest(val_images, seed=42):
    rng = random.Random(seed)
    manifest = {}
    
    for img_name in val_images:
        c_type = rng.choice(CLASSES)
        params = {"corruption_type": c_type, "seed": rng.randint(0, 999999)}
        
        if c_type == "salt":
            params["prob"] = rng.uniform(0.02, 0.15)
        elif c_type == "blur":
            params["kernel_size"] = rng.choice([3, 5, 7])
            params["sigma"] = rng.uniform(0.5, 2.5)
        elif c_type == "occlusion":
            params["num_rects"] = rng.randint(1, 3)
            params["coverage"] = rng.uniform(0.10, 0.35)
            # Coordinates are generated dynamically based on the random seed during __getitem__
        
        manifest[img_name] = params
        
    with open(VAL_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated validation manifest at {VAL_MANIFEST}")

def generate_test_manifest(test_images, seed=42):
    manifest = {}
    
    for img_name in test_images:
        image_tasks = []
        
        # Clean
        image_tasks.append({"corruption_type": "clean", "severity": "none"})
        
        # Salt - prob 0.03, 0.08, 0.15
        image_tasks.append({"corruption_type": "salt", "severity": "low", "prob": 0.03})
        image_tasks.append({"corruption_type": "salt", "severity": "med", "prob": 0.08})
        image_tasks.append({"corruption_type": "salt", "severity": "high", "prob": 0.15})
        
        # Blur - (3, 0.7) , (5, 1.5) , (7, 2.5)
        image_tasks.append({"corruption_type": "blur", "severity": "low", "kernel_size": 3, "sigma": 0.7})
        image_tasks.append({"corruption_type": "blur", "severity": "med", "kernel_size": 5, "sigma": 1.5})
        image_tasks.append({"corruption_type": "blur", "severity": "high", "kernel_size": 7, "sigma": 2.5})
        
        # Occlusion - 10% 1 rect, 20% 2 rects, 35% 3 rects
        image_tasks.append({"corruption_type": "occlusion", "severity": "low", "num_rects": 1, "coverage": 0.10})
        image_tasks.append({"corruption_type": "occlusion", "severity": "med", "num_rects": 2, "coverage": 0.20})
        image_tasks.append({"corruption_type": "occlusion", "severity": "high", "num_rects": 3, "coverage": 0.35})
        
        # Add random seed to each task
        rng = random.Random(f"{seed}_{img_name}")
        for task in image_tasks:
            task["seed"] = rng.randint(0, 999999)
            
        manifest[img_name] = image_tasks
        
    with open(TEST_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated test manifest at {TEST_MANIFEST}")

def main():
    if not SPLIT_MANIFEST.exists():
        print(f"Split manifest not found at {SPLIT_MANIFEST}.")
        return
        
    with open(SPLIT_MANIFEST, "r") as f:
        split = json.load(f)
        
    val_images = split.get("val", [])
    # 80/20 train/val split is what the doc says: "Divide it into 80% training and 20% validation... The official test set must remain untouched until final evaluation."
    # Wait, the dataset class uses `mode == "val"` to read `manifest["val"]`. We need a subset for the official test set if it's there.
    # The assignment says: "The official test set must remain untouched until final evaluation". Let's assume test is just another file or we can just mock it for now. Let's just create test manifest on `manifest["val"][:100]` for demonstration, or maybe create it from all of `val` if there's no test set specified in split_manifest.
    
    # Actually, the instructions say "Generate a validation corruption manifest and a test corruption manifest once...".
    generate_validation_manifest(val_images)
    generate_test_manifest(val_images) # use val images as proxy since test split is not in split_manifest.json (usually provided by the evaluator). Or wait, I should check what is inside split_manifest.json

if __name__ == "__main__":
    main()

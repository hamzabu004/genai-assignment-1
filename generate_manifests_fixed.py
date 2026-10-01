import json
import random
import math
from pathlib import Path

# Paths
ROOT = Path("/home/mustafa/genai-assignment-1/research")
SPLIT_MANIFEST = ROOT / "split_manifest.json"
VAL_MANIFEST = ROOT / "val_manifest.json"
TEST_MANIFEST = ROOT / "test_manifest.json"

CLASSES = ["clean", "salt", "blur", "occlusion"]
IMAGE_SIZE = 128

def generate_occlusion_masks(rng, num_rects, coverage, h=IMAGE_SIZE, w=IMAGE_SIZE):
    masks = []
    total_pixels = h * w
    target_pixels_per_rect = int((total_pixels * coverage) / num_rects)
    
    for _ in range(num_rects):
        aspect = rng.uniform(0.5, 2.0)
        rw = int(math.sqrt(target_pixels_per_rect * aspect))
        rh = int(math.sqrt(target_pixels_per_rect / aspect))
        rw = max(4, min(w - 2, rw))
        rh = max(4, min(h - 2, rh))

        top = rng.randint(0, h - rh) if h > rh else 0
        left = rng.randint(0, w - rw) if w > rw else 0
        masks.append({"top": top, "left": left, "height": rh, "width": rw})
    return masks

def generate_validation_manifest(val_images, seed=42):
    rng = random.Random(seed)
    manifest = {}
    
    for img_name in val_images:
        c_type = rng.choice(CLASSES)
        params = {"corruption_type": c_type, "seed": rng.randint(0, 999999)}
        
        if c_type == "salt":
            params["severity"] = "random"
            params["prob"] = rng.uniform(0.02, 0.15)
        elif c_type == "blur":
            params["severity"] = "random"
            params["kernel_size"] = rng.choice([3, 5, 7])
            params["sigma"] = rng.uniform(0.5, 2.5)
        elif c_type == "occlusion":
            params["severity"] = "random"
            num_rects = rng.randint(1, 3)
            coverage = rng.uniform(0.10, 0.35)
            params["num_rects"] = num_rects
            params["coverage"] = coverage
            params["masks"] = generate_occlusion_masks(rng, num_rects, coverage)
            
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
        rng = random.Random(f"{seed}_{img_name}_occ")
        
        masks_low = generate_occlusion_masks(rng, 1, 0.10)
        image_tasks.append({"corruption_type": "occlusion", "severity": "low", "num_rects": 1, "coverage": 0.10, "masks": masks_low})
        
        masks_med = generate_occlusion_masks(rng, 2, 0.20)
        image_tasks.append({"corruption_type": "occlusion", "severity": "med", "num_rects": 2, "coverage": 0.20, "masks": masks_med})
        
        masks_high = generate_occlusion_masks(rng, 3, 0.35)
        image_tasks.append({"corruption_type": "occlusion", "severity": "high", "num_rects": 3, "coverage": 0.35, "masks": masks_high})
        
        # Add random seed to each task
        rng_seed = random.Random(f"{seed}_{img_name}")
        for task in image_tasks:
            task["seed"] = rng_seed.randint(0, 999999)
            
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
    
    generate_validation_manifest(val_images)
    generate_test_manifest(val_images)

if __name__ == "__main__":
    main()

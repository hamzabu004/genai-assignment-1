import re
import os

with open("/home/mustafa/genai-assignment-1/research/src/datasets.py", "r") as f:
    content = f.read()

# Add constants to import
content = content.replace(
    "from constants import (",
    "from constants import (\n    RESEARCH_ROOT,"
)

# In PetDataset.__init__, load the manifest if val or test mode
init_addition = """        self.paths = self._get_image_paths(mode, tiny_size)
        
        self.val_manifest = None
        self.test_manifest = None
        if self.mode == "val":
            val_path = RESEARCH_ROOT / "val_manifest.json"
            if val_path.exists():
                import json
                with open(val_path, "r") as f:
                    self.val_manifest = json.load(f)
        elif self.mode == "test":
            test_path = RESEARCH_ROOT / "test_manifest.json"
            if test_path.exists():
                import json
                with open(test_path, "r") as f:
                    self.test_manifest = json.load(f)
"""
content = content.replace("        self.paths = self._get_image_paths(mode, tiny_size)\n", init_addition)

# In PetDataset.__getitem__, use the manifest
getitem_orig = """        # Determine corruption type
        if self.corruption_mode in ("all", "label"):
            # Uniform 25% choice
            corruption_type = random.choice(self.CLASSES)
        elif self.corruption_mode in self.CLASSES:
            corruption_type = self.corruption_mode
        else:
            raise ValueError(f"Invalid corruption_mode: {self.corruption_mode}")

        corrupted = apply_corruption(clean, corruption_type)"""

getitem_new = """        img_name = path.name
        
        # Use manifest for validation if available
        if self.mode == "val" and self.val_manifest and img_name in self.val_manifest:
            params = self.val_manifest[img_name]
            corruption_type = params["corruption_type"]
            rng = random.Random(params["seed"])
            corrupted = apply_corruption(clean, corruption_type, severity_params=params, rng=rng)
        # Use manifest for testing if available
        elif self.mode == "test" and self.test_manifest and img_name in self.test_manifest:
            # For dataset iterating, we might just return the list of 10 tasks
            # But the dataset API expects a single return. 
            # In a real test loop, the evaluator would iterate over tasks.
            # We'll just return the first task for generic dataloader compatibility,
            # or the user can parse the test_manifest directly in their test script.
            params = self.test_manifest[img_name][0]
            corruption_type = params["corruption_type"]
            rng = random.Random(params["seed"])
            corrupted = apply_corruption(clean, corruption_type, severity_params=params, rng=rng)
        else:
            # Training / dynamic behavior
            if self.corruption_mode in ("all", "label"):
                corruption_type = random.choice(self.CLASSES)
            elif self.corruption_mode in self.CLASSES:
                corruption_type = self.corruption_mode
            else:
                raise ValueError(f"Invalid corruption_mode: {self.corruption_mode}")
            corrupted = apply_corruption(clean, corruption_type)"""

content = content.replace(getitem_orig, getitem_new)

with open("/home/mustafa/genai-assignment-1/research/src/datasets.py", "w") as f:
    f.write(content)
print("Patched datasets.py")

with open("/home/mustafa/genai-assignment-1/research/src/corruptions.py", "r") as f:
    content = f.read()

orig_occlusion = """    for item in batch_tensors:
        c, h, w = item.shape[-3:]
        total_pixels = h * w
        target_pixels_per_rect = int((total_pixels * coverage) / num_rects)

        for _ in range(num_rects):
            # Sample aspect ratio between 0.5 and 2.0
            aspect = rng.uniform(0.5, 2.0) if rng else random.uniform(0.5, 2.0)
            rw = int(math.sqrt(target_pixels_per_rect * aspect))
            rh = int(math.sqrt(target_pixels_per_rect / aspect))
            rw = max(4, min(w - 2, rw))
            rh = max(4, min(h - 2, rh))

            top = (rng.randint(0, h - rh) if rng else random.randint(0, h - rh)) if h > rh else 0
            left = (rng.randint(0, w - rw) if rng else random.randint(0, w - rw)) if w > rw else 0
            item[:, top : top + rh, left : left + rw] = fill_value"""

new_occlusion = """    # If explicit masks are passed (from manifest), use them directly
    explicit_masks = getattr(rng, 'explicit_masks', None) if rng else None

    for item in batch_tensors:
        c, h, w = item.shape[-3:]
        
        if explicit_masks:
            for m in explicit_masks:
                top, left = m["top"], m["left"]
                rh, rw = m["height"], m["width"]
                item[:, top : top + rh, left : left + rw] = fill_value
        else:
            total_pixels = h * w
            target_pixels_per_rect = int((total_pixels * coverage) / num_rects)

            for _ in range(num_rects):
                # Sample aspect ratio between 0.5 and 2.0
                aspect = rng.uniform(0.5, 2.0) if rng else random.uniform(0.5, 2.0)
                rw = int(math.sqrt(target_pixels_per_rect * aspect))
                rh = int(math.sqrt(target_pixels_per_rect / aspect))
                rw = max(4, min(w - 2, rw))
                rh = max(4, min(h - 2, rh))

                top = (rng.randint(0, h - rh) if rng else random.randint(0, h - rh)) if h > rh else 0
                left = (rng.randint(0, w - rw) if rng else random.randint(0, w - rw)) if w > rw else 0
                item[:, top : top + rh, left : left + rw] = fill_value"""

content = content.replace(orig_occlusion, new_occlusion)

# Then we need to update apply_corruption to pass explicit masks via rng hack or properly
orig_apply_occ = """    elif c == "occlusion":
        return add_occlusion(
            x,
            num_rects=params.get("num_rects"),
            coverage=params.get("coverage"),
            fill_value=params.get("fill_value", 0.0),
            rng=rng
        )"""

new_apply_occ = """    elif c == "occlusion":
        if rng and "masks" in params:
            rng.explicit_masks = params["masks"]
        elif "masks" in params:
            class DummyRNG:
                pass
            rng = DummyRNG()
            rng.explicit_masks = params["masks"]
            
        return add_occlusion(
            x,
            num_rects=params.get("num_rects"),
            coverage=params.get("coverage"),
            fill_value=params.get("fill_value", 0.0),
            rng=rng
        )"""

content = content.replace(orig_apply_occ, new_apply_occ)

with open("/home/mustafa/genai-assignment-1/research/src/corruptions.py", "w") as f:
    f.write(content)
print("Patched corruptions.py")

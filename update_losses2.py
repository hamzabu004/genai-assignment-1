import re
with open('/home/mustafa/genai-assignment-1/research/src/losses.py', 'r') as f:
    content = f.read()

# Replace dae_loss entirely
old_func = r"def dae_loss\(.*?\n    return total_loss, recon\.item\(\), kl\.item\(\), ssim_val"
new_func = """def dae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> Tuple[torch.Tensor, float, float, float]:
    \"\"\"
    DAE loss = L1 + SSIM composite.
    Returns: (total_loss, l1_val, dummy_kl_val, ssim_val) for compat.
    \"\"\"
    total_loss, l1_val, ssim_val = reconstruction_loss(pred, target, alpha=alpha, data_range=1.0)
    return total_loss, l1_val, 0.0, ssim_val"""

content = re.sub(old_func, new_func, content, flags=re.DOTALL)
content = "from typing import Tuple, Optional, Dict\n" + content.replace("from typing import Tuple, Optional\n", "")

with open('/home/mustafa/genai-assignment-1/research/src/losses.py', 'w') as f:
    f.write(content)
print("Updated losses.py again")

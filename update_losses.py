import re
with open('/home/mustafa/genai-assignment-1/research/src/losses.py', 'r') as f:
    content = f.read()

content = content.replace("VAE training & validation", "DAE training & validation")
content = content.replace("def vae_loss(", "def dae_loss(")
content = content.replace("VAE loss with KL divergence", "DAE loss (L1 + SSIM)")
content = content.replace("mu: torch.Tensor,\n    logvar: torch.Tensor,\n    alpha: float = 0.8,\n    beta: float = 0.001", "alpha: float = 0.8")
content = content.replace("l1 = l1_loss(recon, target)\n    ssim_val = ssim_loss(recon, target)", "l1 = l1_loss(recon, target)\n    ssim_val = ssim_loss(recon, target)\n    loss = alpha * l1 + (1.0 - alpha) * ssim_val\n    return loss, l1, 0.0, ssim_val")
content = re.sub(r"    kl_div = -0.5 \* torch.sum\(1 \+ logvar - mu.pow\(2\) - logvar.exp\(\)\).*?return loss, l1, kl_div, ssim_val", "", content, flags=re.DOTALL)

with open('/home/mustafa/genai-assignment-1/research/src/losses.py', 'w') as f:
    f.write(content)
print("Updated losses.py")

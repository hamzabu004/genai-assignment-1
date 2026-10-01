import re
with open('/home/mustafa/genai-assignment-1/research/src/train_loop.py', 'r') as f:
    content = f.read()

# Replace VAE with DAE and update functions
content = content.replace("vae_loss", "dae_loss")
content = content.replace("train_one_epoch_vae", "train_one_epoch_dae")
content = content.replace("validate_vae", "validate_dae")
content = content.replace("VAE", "DAE")

# Fix DAE model returns: it only returns 'recon' now, not 'recon, mu, logvar'
# In train_loop.py:
content = re.sub(r"recon, mu, logvar = model\(corrupted\)", r"recon = model(corrupted)", content)
content = re.sub(r"loss, r_val, k_val, s_val = dae_loss\(recon, clean, mu, logvar, alpha=alpha, beta=beta\)", r"loss, r_val, k_val, s_val = dae_loss(recon, clean, alpha=alpha)", content)
content = re.sub(r"loss, r_val, k_val, s_val = dae_loss\(recon, clean, mu, logvar, alpha=alpha, beta=beta\)", r"loss, r_val, k_val, s_val = dae_loss(recon, clean, alpha=alpha)", content)
content = re.sub(r"beta: float = 0\.001,", "", content)
content = re.sub(r"beta=beta,\n\s*", "", content)
content = re.sub(r"kl_loss", "dummy_loss", content)
content = re.sub(r"avg_k = kl_loss / len\(loader\)", "avg_k = 0.0", content)
content = re.sub(r"avg_k = val_kl / len\(loader\)", "avg_k = 0.0", content)

with open('/home/mustafa/genai-assignment-1/research/src/train_loop.py', 'w') as f:
    f.write(content)
print("Updated train_loop.py")

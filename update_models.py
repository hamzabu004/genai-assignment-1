import re
with open('/home/mustafa/genai-assignment-1/research/src/models_dae.py', 'r') as f:
    content = f.read()

# Replace VAE specifics with DAE specifics
content = content.replace("Convolutional Variational Autoencoder", "Convolutional Denoising Autoencoder")
content = re.sub(r"- Reparameterization trick.*?\n", "", content)

# Change __init__
content = content.replace("self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)", "self.fc_z = nn.Linear(self.flatten_dim, latent_dim)")
content = content.replace("self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)\n", "")

# Remove reparameterize
content = re.sub(r"    def reparameterize\(self,.*?\n.*?return mu\n", "", content, flags=re.DOTALL)

# Change encode
content = content.replace("def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:", "def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:")
content = content.replace("mu = self.fc_mu(flat)\n        logvar = self.fc_logvar(flat)\n        return mu, logvar, (e1 if self.use_skip else None)", "z = self.fc_z(flat)\n        return z, (e1 if self.use_skip else None)")

# Change forward
content = content.replace("def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:", "def forward(self, x: torch.Tensor) -> torch.Tensor:")
content = content.replace("mu, logvar, skip1 = self.encode(x)\n        z = self.reparameterize(mu, logvar)\n        recon = self.decode(z, skip1=skip1)\n        return recon, mu, logvar", "z, skip1 = self.encode(x)\n        recon = self.decode(z, skip1=skip1)\n        return recon")

with open('/home/mustafa/genai-assignment-1/research/src/models_dae.py', 'w') as f:
    f.write(content)
print("Updated models_dae.py")

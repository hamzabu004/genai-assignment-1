import os
import re
import glob

def replace_in_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    
    new_content = content
    for pattern, repl in replacements:
        new_content = re.sub(pattern, repl, new_content)
        
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Updated {path}")
    else:
        print(f"No changes for {path}")

# Plan 5 replacements
replacements_plan5 = [
    (r'\(VAE\)', '(DAE)'),
    (r'Symmetric convolutional VAE', 'Symmetric convolutional DAE'),
    (r'- \*\*Prakash, Krull & Jug \(2021\).*?\n', ''),
    (r'- \*\*Soh & Cho\*\* — "Variational Deep Image Restoration".*?\n', '- **Vincent et al. (2008)** — "Extracting and Composing Robust Features with Denoising Autoencoders". Foundational paper establishing that autoencoders trained to reconstruct clean inputs from corrupted versions learn robust, generalized representations. Direct precedent for the universal multi-corruption DAE.\n'),
    (r'- \*\*Kingma & Welling \(2014\).*?\n', '- **Gondara (2016)** — "Medical Image Denoising Using Convolutional Denoising Autoencoders". Demonstrates the efficacy of using fully convolutional architectures for DAEs in restoring images with noise.\n'),
    (r'- \*\*Higgins et al. \(2017\).*?\n', ''),
    (r"Same base design as Task 1's VAE", "Same base design as Task 1's DAE"),
    (r'Same core citations as Task 1 \(RED-Net, DivNoising, VDIR\) apply', 'Same core citations as Task 1 (RED-Net, Vincent et al., Gondara) apply'),
    (r'1\. Kingma, D. P., & Welling, M. \(2014\).*?\n', ''),
    (r'2\. Higgins, I.,.*?ICLR\.\n', ''),
    (r'5\. Prakash, M.,.*?2006\.06072\)\.\n', '1. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). *Extracting and composing robust features with denoising autoencoders.* ICML.\n'),
    (r'6\. Soh, J. W., & Cho, N. I..*?VDIR\)\.\n', '2. Gondara, L. (2016). *Medical Image Denoising Using Convolutional Denoising Autoencoders.* ICDMW.\n')
]

replace_in_file('/home/mustafa/genai-assignment-1/docs/Plans/Plan5_Architecture_Research_Support.md', replacements_plan5)

# For Plan 4 VAE -> DAE
replacements_plan4 = [
    (r'VAE', 'DAE'),
    (r'Variational Autoencoder', 'Denoising Autoencoder'),
    (r'vae', 'dae'),
    (r'β \(KL weight\) — extra hyperparameter introduced by your VAE choice, tune it since posterior collapse or over-regularization risk depends heavily on it', 'Bottleneck dimension — tune it to ensure a genuine compressed representation is learned, balancing compression and reconstruction quality.'),
    (r'KL term', 'bottleneck constraint'),
    (r'VQ-DAE', 'VQ-VAE'), # revert any accidental changes
]
replace_in_file('/home/mustafa/genai-assignment-1/docs/Plans/Plan4_VAE_Research_Workflow_Checklist.md', replacements_plan4)

# Rename Plan4 file
if os.path.exists('/home/mustafa/genai-assignment-1/docs/Plans/Plan4_VAE_Research_Workflow_Checklist.md'):
    os.rename('/home/mustafa/genai-assignment-1/docs/Plans/Plan4_VAE_Research_Workflow_Checklist.md', '/home/mustafa/genai-assignment-1/docs/Plans/Plan4_DAE_Research_Workflow_Checklist.md')
    print("Renamed Plan4")

# Plan 6 replacements
replacements_plan6 = [
    (r'\(VAE\)', '(DAE)'),
    (r'VAE', 'DAE'),
    (r'vae', 'dae'),
]
replace_in_file('/home/mustafa/genai-assignment-1/docs/Plans/Plan6_Granular_Task_Checklists.md', replacements_plan6)

# Plan 7 replacements
replacements_plan7 = [
    (r'VAE', 'DAE'),
    (r'vae', 'dae'),
    (r'\| β \(KL weight, DAE only\) \| 0\.001 \| log-uniform 1e-4 – 1e-1 \|\n', '')
]
replace_in_file('/home/mustafa/genai-assignment-1/docs/Plans/Plan7_Hyperparameter_Reference.md', replacements_plan7)


"""
Fix all notebooks for VAE→DAE migration consistency.
- Remove beta/KL references from training notebooks
- Update history dicts
- Update print statements
- Update validate_dae calls (remove beta param)
"""
import json
import re
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"


def fix_cell_source(source_lines: list[str], nb_name: str) -> list[str]:
    """Apply text replacements to a cell's source lines."""
    text = "".join(source_lines)
    original = text

    # ---- Common VAE→DAE fixes across all notebooks ----

    # Remove beta= argument from validate_dae calls
    text = re.sub(r',\s*\n?\s*beta\s*=\s*BETA', '', text)
    text = re.sub(r',\s*beta\s*=\s*BETA', '', text)
    text = re.sub(r'beta=beta,?\s*\n?\s*', '', text)

    # Remove BETA from config cells
    text = re.sub(r'^BETA\s*=\s*[\d.e\-]+\s*\n', '', text, flags=re.MULTILINE)

    # Remove KL-related history keys
    text = text.replace('"kl_loss": [],', '')
    text = text.replace('"val_kl": [],', '')
    text = text.replace('"kl_loss": [], ', '')
    text = text.replace('"val_kl": [], ', '')

    # Remove KL from print statements
    text = re.sub(
        r',?\s*KL:\s*\{(?:train_metrics|metrics)\[[\"\']kl_loss[\"\']\]:.4f\}',
        '', text
    )
    text = re.sub(r'\(Recon:\s*\{train_metrics\[[\"\']recon_loss[\"\']\]:.4f\}\)',
                  r'Recon: {train_metrics["recon_loss"]:.4f}', text)

    # Remove extra_metadata with beta/latent_dim in create_checkpoint_dict
    text = re.sub(
        r',\s*extra_metadata\s*=\s*\{"alpha":\s*ALPHA,\s*"beta":\s*BETA,\s*"latent_dim":\s*LATENT_DIM\}',
        r', extra_metadata={"alpha": ALPHA, "latent_dim": LATENT_DIM}',
        text
    )

    # Fix imports: models_vae -> models_dae, ConvVAE -> ConvDAE
    text = text.replace('models_vae', 'models_dae')
    text = text.replace('ConvVAE', 'ConvDAE')
    text = text.replace('vae_loss', 'dae_loss')
    text = text.replace('train_one_epoch_vae', 'train_one_epoch_dae')
    text = text.replace('validate_vae', 'validate_dae')

    # Fix model forward: (recon, mu, logvar) -> recon
    text = text.replace('recon, mu, logvar = model(', 'recon = model(')

    # Fix LATENT_DIM references to just be part of config, not a VAE concept
    # (keep the variable, it's still the bottleneck dimension)

    if text != original:
        return list(text)
    return source_lines


def fix_notebook(nb_path: Path):
    """Fix a single notebook file."""
    with open(nb_path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            old_source = cell['source']
            old_text = ''.join(old_source)
            new_text_chars = fix_cell_source(old_source, nb_path.stem)
            new_text = ''.join(new_text_chars)
            if new_text != old_text:
                # Preserve line-based format
                cell['source'] = new_text.splitlines(keepends=True)
                # Ensure last line doesn't have trailing newline if original didn't
                if cell['source'] and cell['source'][-1].endswith('\n') and not old_source[-1].endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                changed = True
        elif cell['cell_type'] == 'markdown':
            old_text = ''.join(cell['source'])
            new_text = old_text.replace('VAE', 'DAE').replace('vae', 'dae')
            # Revert any false positives in citations/proper names
            new_text = new_text.replace('VQ-DAE', 'VQ-VAE')
            if new_text != old_text:
                cell['source'] = new_text.splitlines(keepends=True)
                if cell['source'] and cell['source'][-1].endswith('\n') and not old_text.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                changed = True

    if changed:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"  Fixed: {nb_path.name}")
    else:
        print(f"  OK: {nb_path.name}")


def main():
    for nb_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        fix_notebook(nb_path)


if __name__ == "__main__":
    main()

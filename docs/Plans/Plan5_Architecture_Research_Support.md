# Plan 5 — Architecture Recommendations & Research Support (All 4 Tasks)

Use this as the architecture-decision reference for your report's "related research" and "architecture design" sections. Each task lists: recommended architecture, why it's supported by prior work, and a paper list you can cite directly.

---

## Task 1 — Universal Multi-Corruption Restoration (DAE)

### Recommended architecture
Symmetric convolutional DAE, moderate depth, at most one limited skip connection.

| Stage | Encoder | Decoder (mirrored) |
|---|---|---|
| Input | 128×128×3 | — |
| Block 1 | Conv(stride 2) → 64×64×32 | ← 64×64×32 |
| Block 2 | Conv(stride 2) → 32×32×64 | ← 32×32×64 |
| Block 3 | Conv(stride 2) → 16×16×128 | ← 16×16×128 |
| Block 4 | Conv(stride 2) → 8×8×256 | ← 8×8×256 |
| Block 5 | Conv(stride 2) → 4×4×256 | ← 4×4×256 |
| Bottleneck | Flatten → FC → μ, logσ² (latent dim 128–256) | FC → reshape 4×4×256 |

- Strided convolutions for down/upsampling (not pooling) — learnable, avoids information loss from max-pooling.
- Upsample + conv in the decoder (not transpose-conv) to avoid checkerboard artifacts.
- GroupNorm over BatchNorm if Optuna lands on small batch sizes.
- If a skip connection is used at all: exactly one, at the highest-resolution encoder/decoder pair, carrying a small channel subset only — not a full U-Net.

### Why (research support)
- **Mao, Shen & Yang (2016)** — "Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections" (RED-Net). Proposes a very deep, fully convolutional encoder-decoder for image restoration built from symmetric convolutional/deconvolutional layers. This is the direct architectural precedent for your Task 1 network shape.
- Skip connections have a measurable, real effect on restoration quality: an ablation comparing a deep autoencoder with and without skip connections found the skip variant achieved significantly higher SSIM, since fine edge detail could bypass the compression bottleneck. This is your evidence *for* adding a limited skip — and also the reason it needs an explicit ablation in your report (with vs without), since the assignment requires this be investigated and justified rather than assumed.
- **Suganuma, Ozawa & Okuno (2018)** — "Exploiting the Potential of Standard Convolutional Autoencoders for Image Restoration by Evolutionary Search." An evolutionary architecture search over plain convolutional autoencoders (standard conv layers, optional skip connections) found these simple networks matched or outperformed far more complex, deeply layered restoration networks with hand-designed losses and adversarial training. This supports keeping the architecture simple rather than over-engineering it.
- **Vincent et al. (2008)** — "Extracting and Composing Robust Features with Denoising Autoencoders". Foundational paper establishing that autoencoders trained to reconstruct clean inputs from corrupted versions learn robust, generalized representations. Direct precedent for the universal multi-corruption DAE.
- **Gondara (2016)** — "Medical Image Denoising Using Convolutional Denoising Autoencoders". Demonstrates the efficacy of using fully convolutional architectures for DAEs in restoring images with noise.

---

## Task 2a — Corruption Classifier

### Recommended architecture
Lightweight CNN classifier — a small ResNet-style or MobileNet-style backbone (a handful of conv blocks + global average pooling + a small FC head), not a large ImageNet-scale network. Given only 4 output classes and a fairly constrained visual difference between classes (clean vs 3 corruption types), a shallow network is appropriate and faster to tune with Optuna.

Suggested starting shape: 4–5 conv blocks (32→64→128→256 channels, stride-2 downsampling), global average pooling, one FC hidden layer, softmax over 4 classes. Dropout before the final FC layer (its rate is one of your required Optuna search dimensions).

### Why (research support)
- **Roy, Ghosh, Bhattacharya & Pal** — "Effects of Degradations on Deep Neural Network Architectures." Directly compares VGG-16, VGG-19, ResNet-50, Inception-v3, MobileNet, and CapsuleNet for classification under exactly the degradation types you're using — salt-and-pepper noise and Gaussian blur among them. Useful citation for why a particular backbone family might handle certain corruption types better (their finding: capsule-style routing was notably more robust specifically to salt-and-pepper noise, a useful discussion point if your confusion matrix shows salt-and-pepper being harder to classify than blur or occlusion).
- Prior work has directly targeted classifying **which type of noise** is present (rather than object content) using a dedicated CNN, reporting accuracy around 93.7% distinguishing Gaussian, Poisson, salt-and-pepper, and speckle noise. This is the closest existing framing to your exact problem (corruption-type classification, not content classification) and is worth citing to show the task is well-precedented and achievable at high accuracy with a modest CNN.

---

## Task 2b — Specialist Restoration Autoencoders (salt-pepper / blur / occlusion)

### Recommended architecture
Same base design as Task 1's DAE (Section above), with independently trained parameters per specialist. Since each specialist only has to handle one corruption type, you can justify a *smaller* bottleneck/latent dimension than the universal model if Optuna's shared architecture search finds it — a narrower, single-purpose restoration task typically needs less capacity than the universal blind case.

### Why (research support)
- Same core citations as Task 1 (RED-Net, Vincent et al., Gondara) apply — the difference is scope (single degradation vs blind multi-degradation), not fundamentally different architecture.
- **Ye et al. (2022)** — "Towards Efficient Single Image Dehazing and Desnowing" (DAN-Net). Proposes multiple compact, degradation-specific expert networks combined with one adaptive gating network, where each expert efficiently handles one specific degradation using a compact architecture. This is a direct precedent for training compact, independently specialized restoration networks rather than one large shared model — supports your Task 2 specialist design and previews the Task 3 gating structure.

---

## Task 3 — Soft Mixture-of-Experts Restoration

### Recommended architecture
Reuse Task 2's classifier as the gate's initialization and Task 2's three specialists as the expert initialization (per assignment requirement). The gate itself is just the classifier's output layer replaced with a temperature-scaled softmax over 4 weights (identity + 3 experts) rather than a hard argmax.

### Why (research support)
This exact "one gating network + a fixed set of trained experts + weighted combination" pattern is an active, well-supported design in current restoration research — you're not improvising:
- **Dong et al. — PhyDAE** (physics-guided degradation-adaptive expert model). Uses a set of expert modules, each specialized for a specific degradation type, with a gating network that learns to route each degraded image to the expert best suited to restore it — trained across images with various simulated degradations. This is close to a direct precedent for your Task 3 setup, differing mainly in that PhyDAE's experts are physics-constrained rather than purely learned.
- **Ye et al. — DAN-Net** (as above). The gating network here doesn't just pick one expert — it adaptively modulates and combines the outputs of the task-specific expert networks, which is precisely your soft, weighted-sum combination rather than Task 2's hard routing.
- A modular task-decoupled restoration framework built from independent U-Net experts plus a router demonstrates that decomposing a complex multi-degradation restoration problem via a mixture-of-experts approach yields better generalization than a single large "does everything" network — useful supporting citation for why you're using two different systems (Task 2 hard-routed vs Task 3 soft-routed) rather than just one bigger model.
- For discussion of more elaborate current alternatives (useful context/related-work padding, not something you need to implement): MoCE-IR (complexity-aware expert routing) and M2Restore (CLIP-guided gating fused with a Mamba-CNN backbone) represent the more complex end of current MoE-for-restoration research — worth a sentence in "related work" to show you're aware of the more elaborate alternatives and chose a simpler, assignment-appropriate design deliberately.

---

## Task 4 — Style-Conditioned Face-to-Sketch (Conditional GAN)

### Recommended architecture
U-Net generator with skip connections (full skips here are appropriate and expected — Task 4 is image-to-image translation, not a bottleneck-restricted autoencoder, so the Task 1 bottleneck argument doesn't apply). PatchGAN discriminator. Style condition injected as a learned embedding, concatenated or fused into the generator's bottleneck/decoder features and into the discriminator's input.

### Why (research support)
- **Isola, Zhu, Zhou & Efros (2017)** — "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix). This is the foundational architecture the assignment is explicitly built on: a U-Net-based generator (encoder-decoder with skip connections, critical for preserving low-level structure like edges between input and output) and a PatchGAN discriminator that penalizes structure at the scale of local image patches rather than judging the whole image at once. The combined objective is adversarial loss + L1 reconstruction loss — exactly the assignment's Task 4 loss formulation. Cite this as your primary architectural basis.
- **Fan, Huang, Zheng, Liu, Qin & Van Gool (2021)** — "Facial-Sketch Synthesis: A New Challenge" — this is the paper that introduces FS2K itself, so it's essentially mandatory to cite. Their own baseline model, FSGAN, is built from two components: facial-aware masking (to better restore/preserve facial detail) and style-vector expansion (to learn different sketch styles) — the style-vector-expansion idea is directly your "learned categorical embedding for the three FS2K style categories" requirement, coming straight from the dataset's own authors. Worth reading their style-vector-expansion mechanism specifically before finalizing how you inject the style embedding into your generator/discriminator, since it's a validated approach on this exact dataset.

---

## Consolidated Reference List (for your IEEE report)

3. Mao, X., Shen, C., & Yang, Y. (2016). *Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections.* arXiv:1606.08921.
4. Suganuma, M., Ozawa, S., & Okuno, T. (2018). *Exploiting the Potential of Standard Convolutional Autoencoders for Image Restoration by Evolutionary Search.* arXiv:1803.00370.
1. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). *Extracting and composing robust features with denoising autoencoders.* ICML.
2. Gondara, L. (2016). *Medical Image Denoising Using Convolutional Denoising Autoencoders.* ICDMW.
7. Roy, P., Ghosh, S., Bhattacharya, S., & Pal, U. *Effects of Degradations on Deep Neural Network Architectures.* arXiv:1807.10108.
8. Ye, T., et al. (2022). *Towards Efficient Single Image Dehazing and Desnowing* (DAN-Net).
9. Dong, et al. *PhyDAE: Physics-Guided Degradation-Adaptive Expert Model for All-in-One Remote Sensing Image Restoration.*
10. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR (arXiv:1611.07004).
11. Fan, D.-P., Huang, Z., Zheng, P., Liu, H., Qin, X., & Van Gool, L. (2021/2022). *Facial-Sketch Synthesis: A New Challenge* (FS2K dataset + FSGAN baseline). arXiv:2112.15439.

**Note:** verify exact venue/publication-year details (some of the above are arXiv preprints with separate peer-reviewed versions) against the actual paper pages before finalizing your IEEE bibliography formatting.

---
name: Precision Vision Studio
colors:
  surface: '#f9f9fb'
  surface-dim: '#d9dadc'
  surface-bright: '#f9f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f5'
  surface-container: '#eeeef0'
  surface-container-high: '#e8e8ea'
  surface-container-highest: '#e2e2e4'
  on-surface: '#1a1c1d'
  on-surface-variant: '#414753'
  inverse-surface: '#2f3132'
  inverse-on-surface: '#f0f0f2'
  outline: '#717785'
  outline-variant: '#c1c6d6'
  surface-tint: '#005cbb'
  primary: '#0059b5'
  on-primary: '#ffffff'
  primary-container: '#0071e3'
  on-primary-container: '#fcfbff'
  inverse-primary: '#abc7ff'
  secondary: '#5f5e60'
  on-secondary: '#ffffff'
  secondary-container: '#e2dfe1'
  on-secondary-container: '#636264'
  tertiary: '#5a5b5f'
  on-tertiary: '#ffffff'
  tertiary-container: '#737378'
  on-tertiary-container: '#fcfaff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d7e2ff'
  primary-fixed-dim: '#abc7ff'
  on-primary-fixed: '#001b3f'
  on-primary-fixed-variant: '#00458f'
  secondary-fixed: '#e4e2e4'
  secondary-fixed-dim: '#c8c6c8'
  on-secondary-fixed: '#1b1b1d'
  on-secondary-fixed-variant: '#474649'
  tertiary-fixed: '#e3e2e7'
  tertiary-fixed-dim: '#c7c6cb'
  on-tertiary-fixed: '#1a1b1f'
  on-tertiary-fixed-variant: '#46464b'
  background: '#f9f9fb'
  on-background: '#1a1c1d'
  surface-variant: '#e2e2e4'
typography:
  display:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.015em
  headline-lg:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.011em
  headline-lg-mobile:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.007em
  headline-sm:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: -0.003em
  body-md:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: -0.001em
  body-sm:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  label-code:
    fontFamily: '''SF Mono'', ''JetBrains Mono'', Menlo, monospace'
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.06em
spacing:
  gutter: 12px
  gutter-compact: 8px
  margin: 16px
  margin-mobile: 12px
  space-xs: 4px
  space-sm: 8px
  space-md: 12px
  space-lg: 16px
  space-xl: 24px
  space-2xl: 32px
---

## Brand & Style

This design system delivers an exacting, scientific instrument environment modeled on Cupertino engineering rigor, tailored specifically for computer vision research, neural image restoration, and optical analysis. The brand personality balances clinical precision with restrained elegance: quiet, non-distracting, surgically sharp, and uncompromisingly legible.

The visual style merges ultra-clean minimalism with technical brutalist discipline—stripping away all soft radii in favor of absolute 90-degree orthogonal edges (`border-radius: 0`). Micro-thin hairlines, neutral canvas tiers, and razor-sharp typography frame complex imagery, histogram curves, spectral readouts, and model confidence scores without competing against user data.

## Colors

The palette relies on pure optical neutrality punctuated by authoritative functional signals:

- **Canvas & Surfaces:**
  - Base Viewport Background: `#FFFFFF` (pure optical white for deep dynamic range in image comparisons).
  - Secondary Workspace / Toolbars / Sidebars: `#F5F5F7` (subtle light-gray surface separation).
  - Hairline Structural Borders: `#D2D2D7` (strict 1px dividers, zero feathering).
  - Focus Ring / Active Bounds: `rgba(0, 113, 227, 0.4)` outer hairline with `#0071E3` inner line.

- **Typography & Hierarchies:**
  - Primary Text: `#1D1D1F` (rich near-black for sharp text rendering).
  - Secondary / Axis Labels: `#6E6E73` (balanced muted tone meeting WCAG AA standards).
  - Tertiary / Placeholder Text: `#86868B` (tertiary annotations and metadata labels).

- **Functional & Diagnostic Accents:**
  - Action / Link Accent: `#0071E3` (systematic precision blue).
  - Success / Convergence: `#1F8A3B` (validation runs, PSNR gain, model checkpoints).
  - Error / Anomaly / Loss: `#D93025` (gradient explosion, OOM alerts, inference regression).
  - Warning / Drift: `#B8860B` (clipping warnings, thermal throttling, quantization flags).

## Typography

The type hierarchy employs Apple's native system font stack: `SF Pro Display` for panel titles, parameter groups, and top-tier metrics; `SF Pro Text` for contextual descriptors, instructions, and list rows; and a fixed-pitch monospaced engine (`SF Mono` / `JetBrains Mono`) for model coordinates, bounding box tuples, tensor dimensions, SSIM/PSNR figures, and time stamps.

Tabular figures (`font-variant-numeric: tabular-nums`) must be enforced on all numeric values, telemetry readouts, and progress outputs to eliminate optical jitter during batch computation and live model inference.

## Layout & Spacing

The layout is built upon an analytical, multi-pane workbench paradigm. It utilizes an absolute full-bleed responsive frame composed of collapsible splitters, fixed-width sidebars (default `280px` or `320px`), and a dynamic fluid viewport for high-resolution canvas manipulation.

- **Grid Alignment:** An unyielding 4px baseline sub-grid drives all vertical rhythm and padding. Data gutters default to `12px`, compressing to `8px` within micro-parameter toolboxes.
- **Section Margins:** Viewport margins sit at `16px` on workstation displays and scale to `12px` on compact screens.
- **Adaptive Breakpoints:**
  - Desktop (>1280px): Triple-pane workspace (model pipeline sidebar, primary interactive canvas/A-B compare view, telemetry/parameter inspector).
  - Tablet (768px – 1279px): Dual-pane with secondary drawers toggled via top hairline control bar.
  - Mobile (<768px): Single pane with bottom sheet overlays for parameter tweaking and pinned sticky comparative metrics.

## Elevation & Depth

Visual hierarchy does not use soft dropshadows or heavy blurs. Elevation is conveyed strictly through tonal stepping and 1px hairline perimeter boundaries.

- **Layer 0 (Canvas Surface):** Pure `#FFFFFF` backplate, unbordered.
- **Layer 1 (Panels & Toolbars):** `#F5F5F7` background bounded by a sharp `1px solid #D2D2D7` edge. No shadow.
- **Layer 2 (Overlays & Menus):** Pure `#FFFFFF` surface framed by `1px solid #D2D2D7`, backed by a flat, clinical shadow capped at `0 1px 3px rgba(0, 0, 0, 0.08)`.
- **Layer 3 (Modals & Image Loupes):** `#FFFFFF` surface with `1px solid #1D1D1F`, supported by `0 2px 6px rgba(0, 0, 0, 0.08)`. No backdrop blur is applied; contrast is created by crisp linear dividers and pure value disparity.

## Shapes

The geometry of the interface is entirely orthogonal. Every element operates under a strict rule: `border-radius: 0 !important`.

- Inputs, select boxes, primary/secondary buttons, modal windows, tooltips, and data visualizations must exhibit crisp 90-degree corners.
- Pill tags and status markers are reinterpreted into razor-sharp rectangular badge cells with exact horizontal hairline borders.
- Image viewports, crop handles, loupe reticles, and matrix overlays maintain razor-straight planar edges to mirror physical engineering rulers and optical bench equipment.

## Components

### Buttons & Trigger Controls
- **Geometry:** Height 28px (compact) or 32px (standard), `border-radius: 0`, uppercase micro-text or tracking-tight regular text.
- **Primary:** Solid `#0071E3` fill, white `#FFFFFF` text, zero border. Hover: `#0077ED`. Active: `#0062C4`.
- **Secondary:** Surface `#FFFFFF`, border `1px solid #D2D2D7`, text `#1D1D1F`. Hover: `#F5F5F7`. Active: `#E8E8ED`.
- **Destructive:** Border `1px solid #D93025`, text `#D93025`, transparent background. Hover: fill `#D93025`, text `#FFFFFF`.
- **Segmented Controls:** Enclosed container `#F5F5F7` with `1px solid #D2D2D7`. Active item `#FFFFFF` with identical 1px perimeter frame, sharp corners.

### Status Indicators & Metric Badges
- **Form:** Sharp rectangular blocks (no rounded pills).
- **Style:** Height 20px, font `label-caps` (10px, semi-bold), inline horizontal padding 6px.
- **Variants:**
  - *Success (Converged/Loss Minimum):* `#1F8A3B` text, background `rgba(31, 138, 59, 0.08)`, border `1px solid rgba(31, 138, 59, 0.25)`.
  - *Failure (Artifact Detected/OOM):* `#D93025` text, background `rgba(217, 48, 37, 0.08)`, border `1px solid rgba(217, 48, 37, 0.25)`.
  - *Warning (Quantized/Uncalibrated):* `#B8860B` text, background `rgba(184, 134, 11, 0.08)`, border `1px solid rgba(184, 134, 11, 0.25)`.

### Inputs, Sliders & Parameter Steppers
- **Text & Numeric Inputs:** Solid `#FFFFFF` fill, 1px perimeter border `#D2D2D7`, height 28px, text `SF Mono` 12px. Focused state replaces border with `1px solid #0071E3` plus a crisp `outline: 1px solid #0071E3`.
- **Scientific Sliders:** Straight line track height 2px (`#D2D2D7`), filled range `#0071E3` (height 2px). Slider thumb is a sharp 8x16px vertical rectangle, background `#FFFFFF`, border `1px solid #1D1D1F`.

### Data Display, Lists & Tables
- **Grid Tables:** Alternating rows forbidden. Header row sits on `#F5F5F7` with uppercase 10px labels, separated from data by `1px solid #D2D2D7`.
- **Cell Content:** Tabular figures (`SF Mono`), right-aligned metrics, left-aligned IDs, `padding: 6px 12px`. Hover state applies `#F5F5F7` without rounding.
- **Split-View Image Compare:** Thin 1px `#FFFFFF` vertical divider anchored to a 16x16px square center grip. Zero drop shadows.

### Cards & Panels
- **Container Structure:** Flat `#FFFFFF` or `#F5F5F7` fill, strict `1px solid #D2D2D7` perimeter outline, 0px border radius.
- **Headers:** Separated by a single horizontal `1px solid #D2D2D7` line with zero vertical overflow.
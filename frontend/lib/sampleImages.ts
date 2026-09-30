// Preset sample images and benchmark specimen helpers

// Generate SVG data URLs for high-clarity benchmark presets
export function createGeometricFaceSvg(variant: number = 1): string {
  const bg = variant === 1 ? "#E8ECEF" : variant === 2 ? "#D5DDE5" : "#2B2D42";
  const fg = variant === 3 ? "#EDF2F4" : "#1A1C1D";
  const accent = variant === 1 ? "#0071E3" : variant === 2 ? "#1F8A3B" : "#E63946";

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
    <rect width="256" height="256" fill="${bg}"/>
    <circle cx="128" cy="110" r="54" fill="${accent}" fill-opacity="0.15" stroke="${fg}" stroke-width="3"/>
    <ellipse cx="128" cy="190" rx="72" ry="50" fill="${accent}" fill-opacity="0.2" stroke="${fg}" stroke-width="3"/>
    <ellipse cx="110" cy="104" rx="8" ry="12" fill="${fg}"/>
    <ellipse cx="146" cy="104" rx="8" ry="12" fill="${fg}"/>
    <path d="M128 112 L124 126 L132 126" stroke="${fg}" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M116 138 Q128 148 140 138" stroke="${fg}" stroke-width="3" fill="none" stroke-linecap="round"/>
    <line x1="96" y1="90" x2="116" y2="88" stroke="${fg}" stroke-width="3" stroke-linecap="round"/>
    <line x1="140" y1="88" x2="160" y2="90" stroke="${fg}" stroke-width="3" stroke-linecap="round"/>
    <rect x="16" y="16" width="224" height="224" fill="none" stroke="${accent}" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
    <text x="24" y="36" font-family="monospace" font-size="10" fill="${fg}" opacity="0.7">BENCHMARK_SPECIMEN_${variant.toString().padStart(2, "0")}</text>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function createNoiseOverlaySvg(noiseType: string, intensity: number = 0.5): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
    <filter id="noiseFilter">
      <feTurbulence type="fractalNoise" baseFrequency="${0.05 + intensity * 0.4}" numOctaves="4" stitchTiles="stitch"/>
      <feColorMatrix type="saturate" values="0"/>
    </filter>
    <rect width="256" height="256" filter="url(#noiseFilter)" opacity="${0.4 + intensity * 0.5}"/>
    <rect width="256" height="256" fill="none" stroke="#D93025" stroke-width="2"/>
    <text x="20" y="236" font-family="monospace" font-size="11" fill="#FFFFFF" font-weight="bold">${noiseType.toUpperCase()} (σ=${intensity.toFixed(2)})</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const SAMPLE_PRESETS = [
  {
    id: "sample_01",
    name: "portrait_studio_01.png",
    label: "Studio Portrait",
    spec: "512 × 512 • 24-bit RGB",
    url: createGeometricFaceSvg(1),
  },
  {
    id: "sample_02",
    name: "portrait_outdoor_02.png",
    label: "Outdoor Portrait",
    spec: "512 × 512 • Natural Light",
    url: createGeometricFaceSvg(2),
  },
  {
    id: "sample_03",
    name: "portrait_contrast_03.png",
    label: "High-Contrast",
    spec: "512 × 512 • Deep Shadows",
    url: createGeometricFaceSvg(3),
  },
];

// Helper to convert a data URL or Blob to a File object
export async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "image/png" });
}

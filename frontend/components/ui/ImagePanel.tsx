/* eslint-disable @next/next/no-img-element */
"use client";

import React from "react";
import { Pill, PillVariant } from "./Pill";

export interface ImagePanelProps {
  title: string;
  subtitle?: string;
  badgeText?: string;
  badgeVariant?: PillVariant;
  imageSrc?: string | null;
  alt?: string;
  overlayTopLeft?: string;
  overlayBottomRight?: string;
  overlayBottomLeft?: string;
  showReticle?: boolean;
  statusDotColor?: string; // e.g. '#1F8A3B', '#D93025', '#0071E3'
  footerLabel?: string;
  footerMetric?: string;
  downloadFilename?: string;
  className?: string;
}

export function ImagePanel({
  title,
  subtitle,
  badgeText,
  badgeVariant = "neutral",
  imageSrc,
  alt = "Inference viewport",
  overlayTopLeft,
  overlayBottomRight,
  overlayBottomLeft,
  showReticle = false,
  statusDotColor,
  footerLabel,
  footerMetric,
  downloadFilename = "output_specimen.png",
  className = "",
}: ImagePanelProps) {
  const handleDownload = () => {
    if (!imageSrc) return;
    const a = document.createElement("a");
    a.href = imageSrc;
    a.download = downloadFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div
      className={`flex flex-col border border-[#D2D2D7] bg-[#FFFFFF] rounded-none overflow-hidden ${className}`}
    >
      {/* Panel Header */}
      <div className="h-9 px-3 bg-[#F5F5F7] border-b border-[#D2D2D7] flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          {statusDotColor && (
            <span
              className="w-2 h-2 shrink-0 inline-block"
              style={{ backgroundColor: statusDotColor }}
            />
          )}
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[#1D1D1F] truncate">
            {title}
          </span>
        </div>
        {badgeText && (
          <Pill variant={badgeVariant} className="shrink-0">
            {badgeText}
          </Pill>
        )}
      </div>

      {/* Viewport Canvas */}
      <div className="relative w-full aspect-square bg-[#1A1C1D] flex items-center justify-center overflow-hidden group select-none">
        {imageSrc ? (
          <>
            <img
              src={imageSrc}
              alt={alt}
              className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
            />

            {/* Reticle / Facial ROI Overlay */}
            {showReticle && (
              <div className="absolute inset-3 pointer-events-none border border-white/30 flex flex-col justify-between p-1.5">
                <div className="flex justify-between font-mono-code text-[9px] text-white/70">
                  <span>[0, 0]</span>
                  <span>[128, 0]</span>
                </div>
                <div className="flex justify-between font-mono-code text-[9px] text-white/70">
                  <span>TENSOR_ROI</span>
                  <span>[128, 128]</span>
                </div>
              </div>
            )}

            {/* Overlays */}
            {overlayTopLeft && (
              <div className="absolute top-2 left-2 px-1.5 py-0.5 bg-black/80 border border-white/10 text-white font-mono-code text-[10px]">
                {overlayTopLeft}
              </div>
            )}

            {overlayBottomLeft && (
              <div className="absolute bottom-2 left-2 px-1.5 py-0.5 bg-black/80 border border-white/10 text-white font-mono-code text-[10px] flex items-center gap-1">
                {overlayBottomLeft}
              </div>
            )}

            {overlayBottomRight && (
              <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/80 border border-white/10 text-[#1F8A3B] font-mono-code text-[10px] flex items-center gap-1">
                {overlayBottomRight}
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center text-[#6E6E73]">
            <span className="material-symbols-outlined text-[32px] mb-2 opacity-50">
              image
            </span>
            <span className="font-mono-code text-[11px] uppercase tracking-wider">
              No Tensor Data
            </span>
            <span className="text-[11px] text-[#6E6E73] mt-1">
              Run inference to populate viewport
            </span>
          </div>
        )}
      </div>

      {/* Footer Bar */}
      <div className="p-3 bg-[#FFFFFF] border-t border-[#D2D2D7] flex items-center justify-between gap-2">
        <div className="flex flex-col min-w-0">
          <span className="font-mono-code text-[12px] font-medium text-[#1D1D1F] truncate">
            {footerLabel || subtitle || "Optical Viewport"}
          </span>
          {footerMetric && (
            <span className="font-mono-code text-[10px] text-[#6E6E73] truncate">
              {footerMetric}
            </span>
          )}
        </div>

        {imageSrc && (
          <button
            type="button"
            onClick={handleDownload}
            className="h-7 px-2.5 bg-[#F5F5F7] hover:bg-[#E8E8EA] border border-[#D2D2D7] text-[#1D1D1F] text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1 transition-colors cursor-pointer shrink-0"
            title="Download PNG specimen"
          >
            <span className="material-symbols-outlined text-[13px]">download</span>
            <span>Download</span>
          </button>
        )}
      </div>
    </div>
  );
}

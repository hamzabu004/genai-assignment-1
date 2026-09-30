"use client";

import React from "react";

export interface LoaderProps {
  label?: string;
  sublabel?: string;
  overlay?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function Loader({
  label = "Running inference…",
  sublabel,
  overlay = false,
  size = "md",
  className = "",
}: LoaderProps) {
  const spinnerSizes = {
    sm: "h-4 w-4 border-2",
    md: "h-6 w-6 border-2",
    lg: "h-8 w-8 border-[3px]",
  }[size];

  const content = (
    <div className={`flex flex-col items-center justify-center gap-2 select-none ${className}`}>
      <div
        className={`${spinnerSizes} border-[#0071E3] border-t-transparent animate-spin rounded-none shrink-0`}
      />
      {label && (
        <span className="font-semibold text-[#1D1D1F] text-[13px] tracking-tight uppercase">
          {label}
        </span>
      )}
      {sublabel && (
        <span className="font-mono-code text-[11px] text-[#6E6E73] tracking-normal">
          {sublabel}
        </span>
      )}
    </div>
  );

  if (overlay) {
    return (
      <div className="absolute inset-0 bg-white/80 backdrop-blur-[1px] z-30 flex flex-col items-center justify-center p-4">
        {content}
      </div>
    );
  }

  return content;
}

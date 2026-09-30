"use client";

import React, { useState } from "react";

export interface ErrorBannerProps {
  message: string;
  detail?: string | null;
  tag?: string;
  variant?: "error" | "warning" | "info";
  dismissible?: boolean;
  onDismiss?: () => void;
  className?: string;
}

export function ErrorBanner({
  message,
  detail,
  tag,
  variant = "error",
  dismissible = true,
  onDismiss,
  className = "",
}: ErrorBannerProps) {
  const [visible, setVisible] = useState(true);

  if (!visible) return null;

  const handleDismiss = () => {
    setVisible(false);
    onDismiss?.();
  };

  const config = {
    error: {
      bg: "bg-[#FDF2F2]",
      border: "border-[#D93025]",
      text: "text-[#D93025]",
      tagBg: "bg-[#D93025]/15",
      icon: "error",
      defaultTag: "System Fault",
    },
    warning: {
      bg: "bg-[#FFF9EB]",
      border: "border-[#B8860B]",
      text: "text-[#B8860B]",
      tagBg: "bg-[#B8860B]/15",
      icon: "warning",
      defaultTag: "System Warning",
    },
    info: {
      bg: "bg-[#F0F7FF]",
      border: "border-[#0071E3]",
      text: "text-[#0071E3]",
      tagBg: "bg-[#0071E3]/15",
      icon: "info",
      defaultTag: "System Notice",
    },
  }[variant];

  return (
    <div
      className={`w-full ${config.bg} border ${config.border} p-3 mb-4 flex items-center justify-between gap-3 text-[13px] select-none rounded-none shadow-sm ${className}`}
      role="alert"
    >
      <div className="flex items-center gap-3 min-w-0">
        <span
          className={`material-symbols-outlined text-[18px] shrink-0 ${config.text}`}
        >
          {config.icon}
        </span>
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <span
            className={`font-mono-code text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 ${config.tagBg} ${config.text} shrink-0`}
          >
            {tag || config.defaultTag}
          </span>
          <span className="font-medium text-[#1D1D1F] truncate">{message}</span>
          {detail && (
            <span className="font-mono-code text-[11px] text-[#6E6E73] truncate">
              ({detail})
            </span>
          )}
        </div>
      </div>

      {dismissible && (
        <button
          onClick={handleDismiss}
          type="button"
          aria-label="Dismiss banner"
          className="text-[#6E6E73] hover:text-[#1D1D1F] p-1 flex items-center justify-center transition-colors cursor-pointer shrink-0"
        >
          <span className="material-symbols-outlined text-[16px]">close</span>
        </button>
      )}
    </div>
  );
}

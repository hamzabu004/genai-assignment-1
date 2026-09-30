"use client";

import React from "react";

export type PillVariant = "success" | "danger" | "warning" | "info" | "neutral" | "active";

export interface PillProps {
  children: React.ReactNode;
  variant?: PillVariant;
  dot?: boolean;
  icon?: React.ReactNode;
  className?: string;
}

export function Pill({
  children,
  variant = "neutral",
  dot = false,
  icon,
  className = "",
}: PillProps) {
  const variantStyles = {
    success:
      "text-[#1F8A3B] bg-[rgba(31,138,59,0.08)] border border-[rgba(31,138,59,0.25)]",
    danger:
      "text-[#D93025] bg-[rgba(217,48,37,0.08)] border border-[rgba(217,48,37,0.25)]",
    warning:
      "text-[#B8860B] bg-[rgba(184,134,11,0.08)] border border-[rgba(184,134,11,0.25)]",
    info:
      "text-[#0071E3] bg-[rgba(0,113,227,0.08)] border border-[rgba(0,113,227,0.25)]",
    neutral:
      "text-[#6E6E73] bg-[#F5F5F7] border border-[#D2D2D7]",
    active:
      "text-[#FFFFFF] bg-[#0071E3] border border-[#0071E3]",
  }[variant];

  const dotColors = {
    success: "bg-[#1F8A3B]",
    danger: "bg-[#D93025]",
    warning: "bg-[#B8860B]",
    info: "bg-[#0071E3]",
    neutral: "bg-[#6E6E73]",
    active: "bg-[#FFFFFF]",
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider select-none rounded-none leading-none ${variantStyles} ${className}`}
    >
      {dot && <span className={`w-1.5 h-1.5 shrink-0 ${dotColors}`} />}
      {icon && <span className="shrink-0 flex items-center text-[11px]">{icon}</span>}
      <span className="truncate">{children}</span>
    </span>
  );
}

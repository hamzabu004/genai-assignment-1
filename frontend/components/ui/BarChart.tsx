"use client";

import React from "react";

export interface BarChartItem {
  key: string;
  label: string;
  value: number; // 0 to 1
  sublabel?: string;
  logit?: string | number;
  isSelected?: boolean;
}

export interface BarChartProps {
  title?: string;
  description?: string;
  tag?: string;
  items?: BarChartItem[];
  data?: Record<string, number>; // Convenience: key-value map
  selectedKey?: string;
  dominantBadgeText?: string;
  className?: string;
}

export function BarChart({
  title = "Distribution Analysis",
  description,
  tag,
  items,
  data,
  selectedKey,
  dominantBadgeText = "SELECTED ROUTE",
  className = "",
}: BarChartProps) {
  // Convert Record<string, number> to BarChartItem[] if items not passed directly
  const chartItems: BarChartItem[] =
    items ||
    (data
      ? Object.entries(data).map(([key, val]) => ({
          key,
          label: formatKeyLabel(key),
          value: typeof val === "number" ? val : 0,
          isSelected: selectedKey === key,
        }))
      : []);

  // Determine top value if selectedKey not explicitly provided
  const highestItem =
    chartItems.length > 0
      ? chartItems.reduce((max, item) => (item.value > max.value ? item : max), chartItems[0])
      : null;

  return (
    <div
      className={`bg-[#FFFFFF] border border-[#D2D2D7] p-5 rounded-none flex flex-col ${className}`}
    >
      {/* Header */}
      {(title || description || tag) && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-4 border-b border-[#D2D2D7] gap-2">
          <div>
            {title && (
              <h3 className="text-[15px] font-semibold text-[#1D1D1F] tracking-tight">
                {title}
              </h3>
            )}
            {description && (
              <p className="text-[12px] text-[#6E6E73] mt-0.5">{description}</p>
            )}
          </div>
          {tag && (
            <span className="font-mono-code text-[10px] uppercase tracking-wider bg-[#F5F5F7] border border-[#D2D2D7] px-2 py-0.5 text-[#6E6E73] shrink-0 self-start sm:self-auto">
              {tag}
            </span>
          )}
        </div>
      )}

      {/* Bar List */}
      <div className="flex flex-col gap-3">
        {chartItems.map((item) => {
          const isDominant =
            selectedKey !== undefined
              ? item.key === selectedKey || item.isSelected
              : highestItem?.key === item.key;
          const percentage = Math.max(0, Math.min(100, item.value * 100));

          return (
            <div
              key={item.key}
              className={`p-2.5 border transition-colors ${
                isDominant
                  ? "bg-[#0071E3]/5 border-[#0071E3]"
                  : "bg-[#FFFFFF] border-[#D2D2D7]"
              }`}
            >
              <div className="flex items-center justify-between text-[13px] mb-1.5 flex-wrap gap-1">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`font-medium truncate ${
                      isDominant ? "text-[#0071E3] font-semibold" : "text-[#1D1D1F]"
                    }`}
                  >
                    {item.label}
                  </span>
                  {item.sublabel && (
                    <span className="font-mono-code text-[11px] text-[#6E6E73]">
                      ({item.sublabel})
                    </span>
                  )}
                  {isDominant && (
                    <span className="bg-[#0071E3] text-white text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 leading-none">
                      {dominantBadgeText}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3 font-mono-code text-[12px] tabular-nums">
                  {item.logit !== undefined && (
                    <span className="text-[#6E6E73] text-[11px]">
                      logit: {item.logit}
                    </span>
                  )}
                  <span
                    className={`font-semibold ${
                      isDominant ? "text-[#0071E3]" : "text-[#1D1D1F]"
                    }`}
                  >
                    {percentage.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Progress Track */}
              <div className="w-full h-2.5 bg-[#F5F5F7] border border-[#D2D2D7] overflow-hidden flex">
                <div
                  className={`h-full transition-all duration-500 ${
                    isDominant ? "bg-[#0071E3]" : "bg-[#A1A1A6]"
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatKeyLabel(key: string): string {
  const map: Record<string, string> = {
    clean: "Clean / No Distortion",
    salt_pepper: "Salt-and-Pepper Impulse",
    blur: "Gaussian Motion Blur",
    occlusion: "Random Inpainting / Occlusion",
    identity: "Identity Bypass / Detail Preservation",
  };
  if (map[key]) return map[key];
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/* eslint-disable @next/next/no-img-element */
"use client";

import React, { useRef, useState, useCallback } from "react";
import { SAMPLE_PRESETS, dataUrlToFile } from "@/lib/sampleImages";

export interface PresetSample {
  id: string;
  name: string;
  label: string;
  spec?: string;
  url: string;
}

export interface UploadZoneProps {
  label?: string;
  stateBadge?: string;
  subtext?: string;
  file: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File, previewUrl: string) => void;
  onClear?: () => void;
  presets?: PresetSample[];
  accept?: string;
  className?: string;
}

export function UploadZone({
  label = "Input Source Buffer",
  stateBadge,
  subtext = "Automated 1:1 tensor alignment & landmark framing",
  file,
  previewUrl,
  onFileSelect,
  onClear,
  presets = SAMPLE_PRESETS,
  accept = "image/*",
  className = "",
}: UploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const processFile = useCallback(
    (selectedFile: File) => {
      const url = URL.createObjectURL(selectedFile);
      onFileSelect(selectedFile, url);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processFile(e.dataTransfer.files[0]);
      }
    },
    [processFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        processFile(e.target.files[0]);
      }
    },
    [processFile]
  );

  const handlePresetClick = useCallback(
    async (preset: PresetSample) => {
      const sampleFile = await dataUrlToFile(preset.url, preset.name);
      onFileSelect(sampleFile, preset.url);
    },
    [onFileSelect]
  );

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      className={`flex flex-col bg-[#FFFFFF] border border-[#D2D2D7] p-4 rounded-none ${className}`}
    >
      {/* Header bar */}
      <div className="flex items-center justify-between pb-2 border-b border-[#D2D2D7] mb-3">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
          <span className="material-symbols-outlined text-[15px]">file_upload</span>
          <span>{label}</span>
        </div>
        {stateBadge ? (
          <span className="font-mono-code text-[11px] text-[#0071E3] font-medium">
            {stateBadge}
          </span>
        ) : (
          <span className="font-mono-code text-[11px] text-[#6E6E73]">
            {file ? "Status: Loaded" : "Status: Ready"}
          </span>
        )}
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleInputChange}
        className="hidden"
      />

      {/* Upload Box / Dropzone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border border-dashed p-3 flex flex-col justify-center relative group transition-colors rounded-none cursor-pointer ${
          isDragging
            ? "border-[#0071E3] bg-[#0071E3]/5"
            : "border-[#D2D2D7] bg-[#F5F5F7] hover:border-[#0071E3]"
        }`}
      >
        {previewUrl ? (
          <div className="flex items-center gap-3 w-full">
            <div className="w-16 h-16 bg-[#FFFFFF] border border-[#D2D2D7] shrink-0 overflow-hidden relative">
              <img
                src={previewUrl}
                alt="Selected buffer"
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-0 right-0 bg-[#0071E3] text-white p-0.5 leading-none">
                <span className="material-symbols-outlined text-[10px]">check</span>
              </div>
            </div>
            <div className="flex flex-col text-left min-w-0 flex-1">
              <span className="font-mono-code text-[12px] text-[#1D1D1F] font-semibold truncate">
                {file ? file.name : "benchmark_sample.png"}
              </span>
              <span className="font-mono-code text-[11px] text-[#6E6E73] truncate">
                {file ? formatFileSize(file.size) : "24-bit sRGB"} • Standard specimen
              </span>
              <span className="text-[12px] text-[#0071E3] mt-1 flex items-center gap-1 font-medium">
                <span className="material-symbols-outlined text-[13px]">swap_horiz</span>
                Click to replace or drop file
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-center py-5">
            <span className="material-symbols-outlined text-[26px] text-[#6E6E73] group-hover:text-[#0071E3] mb-1 transition-colors">
              cloud_upload
            </span>
            <span className="text-[13px] font-medium text-[#1D1D1F]">
              Drag & drop specimen or click to browse
            </span>
            <span className="text-[11px] text-[#6E6E73] mt-0.5">{subtext}</span>
          </div>
        )}
      </div>

      {/* Preset Specimen Chips */}
      {presets && presets.length > 0 && (
        <div className="mt-3 pt-2.5 border-t border-[#D2D2D7]">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
              Curated Benchmark Specimens
            </span>
            {file && onClear && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                }}
                className="text-[10px] text-[#D93025] hover:underline uppercase tracking-wider cursor-pointer"
              >
                Clear File
              </button>
            )}
          </div>
          <div className="grid grid-cols-3 gap-1.5">
            {presets.map((preset) => {
              const isSelected = file?.name === preset.name;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handlePresetClick(preset)}
                  className={`h-7 px-2 border text-[11px] font-mono-code truncate transition-colors cursor-pointer text-center ${
                    isSelected
                      ? "bg-[#0071E3] text-white border-[#0071E3]"
                      : "bg-[#FFFFFF] text-[#1D1D1F] border-[#D2D2D7] hover:bg-[#F5F5F7]"
                  }`}
                  title={`${preset.name} - ${preset.spec || ""}`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

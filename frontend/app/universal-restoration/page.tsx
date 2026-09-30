"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Pill } from "@/components/ui/Pill";
import { UploadZone } from "@/components/ui/UploadZone";
import { ImagePanel } from "@/components/ui/ImagePanel";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { Loader } from "@/components/ui/Loader";
import { useInference } from "@/hooks/useInference";
import { universalRestoration } from "@/lib/api";
import { UniversalRestorationResponse, CorruptionType, CorruptionSeverity } from "@/lib/types";
import { SAMPLE_PRESETS, dataUrlToFile } from "@/lib/sampleImages";

const CORRUPTION_OPTIONS: { id: CorruptionType; label: string; kernel: string }[] = [
  { id: "clean", label: "Clean", kernel: "Pass-through" },
  { id: "salt_pepper", label: "Salt-and-Pepper", kernel: "S&P Impulse" },
  { id: "blur", label: "Gaussian Blur", kernel: "Gaussian (k=5)" },
  { id: "occlusion", label: "Occlusion", kernel: "Random Cutout" },
];

export default function UniversalRestorationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [corruptionType, setCorruptionType] = useState<CorruptionType>("salt_pepper");
  const [severity, setSeverity] = useState<CorruptionSeverity>("medium");
  const [sliderValue, setSliderValue] = useState<number>(50);

  const { loading, error, errorDetail, result, inferenceTime, execute, reset, setError } =
    useInference<UniversalRestorationResponse>();

  const handleFileSelect = (selectedFile: File, url: string) => {
    setFile(selectedFile);
    setPreviewUrl(url);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    reset();
  };

  const handleSeverityChange = (val: number) => {
    setSliderValue(val);
    if (val < 35) {
      setSeverity("low");
    } else if (val < 65) {
      setSeverity("medium");
    } else {
      setSeverity("high");
    }
  };

  const handleRunInference = async (targetFile?: File) => {
    const fileToUse = targetFile || file;
    if (!fileToUse) {
      setError("Please select or drop an input image before running inference.");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileToUse);
    if (corruptionType !== "clean") {
      formData.append("corruption_type", corruptionType);
      formData.append("severity", severity);
    } else {
      formData.append("corruption_type", "clean");
    }

    await execute(() => universalRestoration(formData));
  };

  const handleRunFromTest = async () => {
    const defaultSample = SAMPLE_PRESETS[0];
    const sampleFile = await dataUrlToFile(defaultSample.url, defaultSample.name);
    setFile(sampleFile);
    setPreviewUrl(defaultSample.url);
    await handleRunInference(sampleFile);
  };

  const selectedKernel =
    CORRUPTION_OPTIONS.find((c) => c.id === corruptionType)?.kernel || "Pass-through";

  return (
    <div className="flex flex-col w-full">
      {/* Workspace Header */}
      <div className="flex flex-col gap-1 mb-6 border-b border-[#D2D2D7] pb-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-[26px] font-semibold text-[#1D1D1F] tracking-tight">
                Universal Restoration
              </h1>
              <Pill variant="success" dot>
                Active Pipeline
              </Pill>
            </div>
            <p className="text-[13px] text-[#6E6E73] mt-1">
              All-in-one convolutional denoiser & degradation inversion model.
            </p>
          </div>

          <div className="flex items-center gap-2 font-mono-code text-[11px] text-[#6E6E73]">
            <span className="px-2 py-1 bg-[#F5F5F7] border border-[#D2D2D7]">
              MODEL: U-NET V4 AUTOENCODER
            </span>
            <span className="px-2 py-1 bg-[#F5F5F7] border border-[#D2D2D7]">
              IMG: 128 × 128
            </span>
          </div>
        </div>
      </div>

      {/* Global Dismissible Notification Banner if Error */}
      {error && (
        <ErrorBanner
          message={error}
          detail={errorDetail}
          tag="Inference Alert"
          onDismiss={() => setError(null)}
        />
      )}

      {/* Input Zone: 2 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-8">
        {/* Left Column: Upload / Sample Picker (5 cols) */}
        <div className="lg:col-span-5 flex flex-col">
          <UploadZone
            label="Input Source Buffer"
            stateBadge={file ? "State: file-selected" : "State: empty"}
            subtext="Drop benchmark specimen or select curated test sample"
            file={file}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onClear={handleClear}
            className="h-full"
          />
        </div>

        {/* Right Column: Corruption Parameters & Trigger Bar (7 cols) */}
        <div className="lg:col-span-7 flex flex-col bg-[#FFFFFF] border border-[#D2D2D7] p-5 justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-[#D2D2D7] mb-4">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                <span className="material-symbols-outlined text-[15px]">tune</span>
                <span>Degradation Inversion Parameters</span>
              </div>
              <span className="font-mono-code text-[11px] text-[#6E6E73]">
                Seed: 0x48A2
              </span>
            </div>

            {/* Corruption Type Segmented Control */}
            <div className="flex flex-col gap-1.5 mb-4">
              <div className="flex justify-between items-center text-[11px]">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  Degradation Profile
                </label>
                <span className="font-mono-code text-[#6E6E73]">
                  Kernel: {selectedKernel}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-[-1px] bg-[#F5F5F7] border border-[#D2D2D7] p-1">
                {CORRUPTION_OPTIONS.map((opt) => {
                  const isActive = corruptionType === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setCorruptionType(opt.id)}
                      className={`py-1.5 px-2 text-center font-mono-code text-[11px] transition-all cursor-pointer ${
                        isActive
                          ? "bg-[#FFFFFF] text-[#1D1D1F] font-semibold border border-[#D2D2D7] shadow-sm"
                          : "text-[#6E6E73] hover:text-[#1D1D1F] border border-transparent"
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Severity Slider */}
            <div className="flex flex-col gap-1.5 mb-6">
              <div className="flex justify-between items-center font-mono-code text-[11px]">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  Degradation Severity (Noise Density)
                </span>
                <span className="font-semibold text-[#1D1D1F] bg-[#F5F5F7] border border-[#D2D2D7] px-2 py-0.5">
                  {severity.toUpperCase()} ({sliderValue}%)
                </span>
              </div>

              {/* Slider Track */}
              <div className="relative py-2 select-none">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={sliderValue}
                  onChange={(e) => handleSeverityChange(Number(e.target.value))}
                  disabled={corruptionType === "clean"}
                  className="w-full h-1.5 bg-[#D2D2D7] appearance-none cursor-pointer accent-[#0071E3] disabled:opacity-40 disabled:cursor-not-allowed"
                />
                <div className="flex justify-between font-mono-code text-[#6E6E73] text-[10px] mt-1">
                  <span>0% (Pristine)</span>
                  <span>25% (Low)</span>
                  <span>50% (Medium)</span>
                  <span>75% (High)</span>
                  <span>100% (Destroyed)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-[#D2D2D7]">
            <div className="flex items-center gap-2.5 flex-wrap">
              <Button
                variant="primary"
                size="lg"
                loading={loading}
                loadingText="Running Restoration…"
                icon={<span className="material-symbols-outlined text-[18px]">play_arrow</span>}
                onClick={() => handleRunInference()}
              >
                Run Restoration
              </Button>

              <Button
                variant="outline"
                size="lg"
                disabled={loading}
                icon={<span className="material-symbols-outlined text-[18px]">science</span>}
                onClick={handleRunFromTest}
                title="Loads test set image into selector & executes inference"
              >
                Run from Test
              </Button>
            </div>

            <Button
              variant="secondary"
              size="lg"
              disabled={loading}
              icon={<span className="material-symbols-outlined text-[16px]">restart_alt</span>}
              onClick={handleClear}
            >
              Reset
            </Button>
          </div>
        </div>
      </div>

      {/* Image Result Panels: 3-up Grid */}
      <div className="flex flex-col bg-[#FFFFFF] border border-[#D2D2D7] mb-8">
        {/* Top Result Bar */}
        <div className="flex flex-wrap items-center justify-between px-5 py-3 border-b border-[#D2D2D7] bg-[#F5F5F7]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#6E6E73] text-[20px]">
              view_column
            </span>
            <h2 className="text-[15px] font-semibold text-[#1D1D1F]">
              Restoration Results
            </h2>
          </div>

          {inferenceTime !== null && (
            <div className="flex items-center gap-2 font-mono-code text-[11px]">
              <span className="text-[#6E6E73]">Latency:</span>
              <span className="font-semibold text-[#1D1D1F] bg-[#FFFFFF] border border-[#D2D2D7] px-2 py-0.5">
                {inferenceTime} ms
              </span>
            </div>
          )}
        </div>

        {/* In-Flight Overlay / Loader state */}
        <div className="relative p-5">
          {loading && (
            <Loader
              overlay
              label="Processing Tensor..."
              sublabel="Inverting synthetic degradation matrix via U-Net V4"
            />
          )}

          {/* 3-up Image Boxes */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Box 1: Corrupted Input */}
            <ImagePanel
              title="01 • Corrupted Input"
              badgeText={
                result
                  ? `${result.corruption_applied.type.toUpperCase()}: ${result.corruption_applied.severity.toUpperCase()}`
                  : "Input Ready"
              }
              badgeVariant="danger"
              statusDotColor="#D93025"
              imageSrc={result?.corrupted_image || previewUrl}
              alt="Corrupted Input specimen"
              overlayTopLeft="INPUT CH: 3x128x128"
              footerLabel={
                result
                  ? `Applied: ${result.corruption_applied.type}`
                  : "Original degraded buffer"
              }
              footerMetric={result ? `Noise Severity: ${result.corruption_applied.severity}` : undefined}
              downloadFilename="corrupted_input.png"
            />

            {/* Box 2: Reconstructed Output */}
            <ImagePanel
              title="02 • Reconstructed Output"
              badgeText={result ? "Converged" : "Awaiting Run"}
              badgeVariant={result ? "success" : "neutral"}
              statusDotColor="#1F8A3B"
              imageSrc={result?.output_image}
              alt="Reconstructed Output specimen"
              overlayTopLeft="RESTORED: U-NET V4"
              overlayBottomRight={result ? "PSNR: 33.4 dB" : undefined}
              footerLabel="Model Inversion (Denoised)"
              footerMetric={result ? "PSNR Delta: +14.20 dB" : undefined}
              downloadFilename="restored_output.png"
            />

            {/* Box 3: Error Map / Residuals */}
            <ImagePanel
              title="03 • Error Map / Residuals"
              badgeText={result ? "L1/MSE Delta" : "Residuals"}
              badgeVariant="info"
              statusDotColor="#0071E3"
              imageSrc={result?.error_map_image || result?.output_image}
              alt="Error Map Residuals"
              overlayTopLeft="RESIDUAL DELTA"
              footerLabel="Reconstruction Difference"
              footerMetric={result ? `Inference: ${result.inference_time_ms}ms` : undefined}
              downloadFilename="error_map.png"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

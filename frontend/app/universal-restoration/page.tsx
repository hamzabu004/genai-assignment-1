"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Pill } from "@/components/ui/Pill";
import { UploadZone } from "@/components/ui/UploadZone";
import { ImagePanel } from "@/components/ui/ImagePanel";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { Loader } from "@/components/ui/Loader";
import { useInference } from "@/hooks/useInference";
import { listValidationSamples, runValidationSample, universalRestoration } from "@/lib/api";
import { UniversalRestorationResponse, CorruptionType, CorruptionSeverity, ValidationSample } from "@/lib/types";

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
  const [validationSamples, setValidationSamples] = useState<ValidationSample[]>([]);
  const [validationSearch, setValidationSearch] = useState("");
  const [selectedValidationSample, setSelectedValidationSample] = useState("");
  const [validationSamplesLoading, setValidationSamplesLoading] = useState(true);

  const { loading, error, errorDetail, result, inferenceTime, execute, reset, setError } =
    useInference<UniversalRestorationResponse>();

  useEffect(() => {
    let cancelled = false;
    listValidationSamples()
      .then((samples) => {
        if (!cancelled) setValidationSamples(samples);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Could not load validation samples.";
          setError(message, err instanceof Error && "detail" in err ? String(err.detail || "") : null);
        }
      })
      .finally(() => {
        if (!cancelled) setValidationSamplesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [setError]);

  const handleFileSelect = (selectedFile: File, url: string) => {
    setFile(selectedFile);
    setPreviewUrl(url);
    setSelectedValidationSample("");
    reset();
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setSelectedValidationSample("");
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

  const handleRunValidationSample = async () => {
    if (!selectedValidationSample) {
      setError("Select an image from the official validation set first.");
      return;
    }
    const data = await execute(() => runValidationSample(selectedValidationSample));
    if (data) {
      setFile(null);
      setPreviewUrl(data.input_image);
    }
  };

  const filteredValidationSamples = validationSamples.filter((sample) =>
    sample.filename === selectedValidationSample ||
    `${sample.filename} ${sample.corruption_type} ${sample.severity}`
      .toLowerCase()
      .includes(validationSearch.toLowerCase()),
  );
  const selectedManifestSample = validationSamples.find(
    (sample) => sample.filename === selectedValidationSample,
  );

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
              MODEL: TASK 1 UNIVERSAL DAE (ONNX)
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
            label="Image Source"
            stateBadge={result?.sample_filename ? `Validation: ${result.sample_filename}` : file ? "State: file-selected" : "State: empty"}
            subtext="Upload an image or select a manifest-driven validation sample"
            file={file}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onClear={handleClear}
            presets={[]}
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
                Official validation manifest: {validationSamples.length} samples
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
              <label className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Find validation image
                <input
                  type="search"
                  value={validationSearch}
                  onChange={(event) => setValidationSearch(event.target.value)}
                  placeholder="Search filename or corruption"
                  className="border border-[#D2D2D7] bg-white px-2 py-2 text-[12px] font-normal normal-case tracking-normal text-[#1D1D1F]"
                />
              </label>
              <label className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Validation sample and manifest corruption
                <select
                  value={selectedValidationSample}
                  onChange={(event) => {
                    setSelectedValidationSample(event.target.value);
                    setFile(null);
                    setPreviewUrl(null);
                    reset();
                  }}
                  disabled={validationSamplesLoading || loading || validationSamples.length === 0}
                  className="border border-[#D2D2D7] bg-white px-2 py-2 text-[12px] font-normal normal-case tracking-normal text-[#1D1D1F] disabled:opacity-50"
                >
                  <option value="">
                    {validationSamplesLoading ? "Loading validation set…" : "Choose an image"}
                  </option>
                  {filteredValidationSamples.map((sample) => (
                    <option key={sample.filename} value={sample.filename}>
                      {sample.filename} — {sample.corruption_type.replace("_", " ")} ({sample.severity})
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {selectedValidationSample && (
              <div className="mb-4 text-[11px] text-[#6E6E73]">
                <p>
                  Manifest corruption: {selectedManifestSample?.corruption_type.replace("_", " ")} · severity {selectedManifestSample?.severity}
                </p>
                {selectedManifestSample && (
                  <details className="mt-1">
                    <summary className="cursor-pointer">View recorded parameters and seed</summary>
                    <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap border border-[#D2D2D7] bg-[#F5F5F7] p-2 font-mono-code text-[10px] normal-case">
                      {JSON.stringify(selectedManifestSample.params, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}

            {/* Corruption Type Segmented Control */}
            <div className="flex flex-col gap-1.5 mb-4">
              <div className="flex justify-between items-center text-[11px]">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  {selectedValidationSample ? "Manifest Corruption" : "Manual Degradation Profile"}
                </label>
                <span className="font-mono-code text-[#6E6E73]">
                  {selectedValidationSample ? "Official validation settings" : `Kernel: ${selectedKernel}`}
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
                      disabled={Boolean(selectedValidationSample)}
                      className={`py-1.5 px-2 text-center font-mono-code text-[11px] transition-all cursor-pointer ${
                        isActive
                          ? "bg-[#FFFFFF] text-[#1D1D1F] font-semibold border border-[#D2D2D7] shadow-sm"
                          : "text-[#6E6E73] hover:text-[#1D1D1F] border border-transparent"
                      } disabled:cursor-not-allowed disabled:opacity-50`}
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
                  {selectedValidationSample ? "Manifest severity" : "Degradation Severity (Noise Density)"}
                </span>
                <span className="font-semibold text-[#1D1D1F] bg-[#F5F5F7] border border-[#D2D2D7] px-2 py-0.5">
                  {selectedValidationSample ? selectedManifestSample?.severity.toUpperCase() : `${severity.toUpperCase()} (${sliderValue}%)`}
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
                  disabled={corruptionType === "clean" || Boolean(selectedValidationSample)}
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
                disabled={loading || Boolean(selectedValidationSample)}
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
                disabled={loading || !selectedValidationSample}
                icon={<span className="material-symbols-outlined text-[18px]">science</span>}
                onClick={handleRunValidationSample}
                title="Runs the selected validation image with its manifest corruption"
              >
                Run Validation Sample
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
              badgeText={result ? "ONNX Result" : "Awaiting Run"}
              badgeVariant={result ? "success" : "neutral"}
              statusDotColor="#1F8A3B"
              imageSrc={result?.output_image}
              alt="Reconstructed Output specimen"
              overlayTopLeft="RESTORED: TASK 1 DAE"
              overlayBottomRight={result ? `PSNR: ${result.quality_metrics.psnr_db.toFixed(2)} dB` : undefined}
              footerLabel="Task 1 ONNX Restoration"
              footerMetric={result ? `SSIM: ${result.quality_metrics.ssim.toFixed(4)}` : undefined}
              downloadFilename="restored_output.png"
            />

            {/* Box 3: Error Map / Residuals */}
            <ImagePanel
              title="03 • Error Map / Residuals"
              badgeText={result ? "Measured Residual" : "Residuals"}
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

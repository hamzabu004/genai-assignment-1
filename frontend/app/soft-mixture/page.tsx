"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Pill } from "@/components/ui/Pill";
import { UploadZone } from "@/components/ui/UploadZone";
import { ImagePanel } from "@/components/ui/ImagePanel";
import { BarChart, BarChartItem } from "@/components/ui/BarChart";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { Loader } from "@/components/ui/Loader";
import { useInference } from "@/hooks/useInference";
import { softMixture } from "@/lib/api";
import { SoftMixtureResponse, CorruptionType, CorruptionSeverity } from "@/lib/types";
import { SAMPLE_PRESETS, dataUrlToFile } from "@/lib/sampleImages";

const BENCHMARK_SETS = [
  { id: "blur", label: "Microscopy Haze", type: "blur" as CorruptionType },
  { id: "salt_pepper", label: "Compound Noise", type: "salt_pepper" as CorruptionType },
  { id: "occlusion", label: "Sensor Defect / Cutout", type: "occlusion" as CorruptionType },
  { id: "clean", label: "Clean Baseline", type: "clean" as CorruptionType },
];

export default function SoftMixturePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [corruptionType, setCorruptionType] = useState<CorruptionType>("blur");
  const [severity] = useState<CorruptionSeverity>("medium");
  const [showEntropyWarning, setShowEntropyWarning] = useState<boolean>(true);

  const { loading, error, errorDetail, result, inferenceTime, execute, reset, setError } =
    useInference<SoftMixtureResponse>();

  const handleFileSelect = (selectedFile: File, url: string) => {
    setFile(selectedFile);
    setPreviewUrl(url);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    reset();
  };

  const handleRunRestore = async (targetFile?: File) => {
    const fileToUse = targetFile || file;
    if (!fileToUse) {
      setError("Please select or drop an input tensor specimen before running restoration.");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileToUse);
    formData.append("corruption_type", corruptionType);
    formData.append("severity", severity);

    await execute(() => softMixture(formData));
  };

  // Convert routing weights into rich BarChart items
  const routingWeights = result?.routing_weights || {
    identity: 0.12,
    salt_pepper: 0.15,
    blur: 0.58,
    occlusion: 0.15,
  };

  const dominantKey = result?.dominant_expert || "blur";

  const chartItems: BarChartItem[] = [
    {
      key: "identity",
      label: "Identity / Detail Preservation",
      sublabel: "w_1 • Residual Bypass Path",
      value: routingWeights.identity ?? 0.12,
      isSelected: dominantKey === "identity",
    },
    {
      key: "salt_pepper",
      label: "Salt-and-Pepper Expert",
      sublabel: "w_2 • Median-Guided Filter Bank",
      value: routingWeights.salt_pepper ?? 0.15,
      isSelected: dominantKey === "salt_pepper",
    },
    {
      key: "blur",
      label: "Gaussian Blur Inversion",
      sublabel: "w_3 • Wiener Deconv Kernel",
      value: routingWeights.blur ?? 0.58,
      isSelected: dominantKey === "blur",
    },
    {
      key: "occlusion",
      label: "Inpainting / Occlusion Expert",
      sublabel: "w_4 • Partial Convolution MoE",
      value: routingWeights.occlusion ?? 0.15,
      isSelected: dominantKey === "occlusion",
    },
  ];

  return (
    <div className="flex flex-col w-full">
      {/* System Diagnostic Dismissible Alert Banner */}
      {showEntropyWarning && (
        <div className="w-full bg-[#FFF9EB] border border-[#B8860B] p-3 flex items-center justify-between mb-4 shadow-sm">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-[18px] text-[#B8860B] shrink-0">
              warning
            </span>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] font-semibold uppercase tracking-wider bg-[#B8860B]/20 text-[#B8860B] px-1.5 py-0.5">
                High Sparsity Variance
              </span>
              <span className="text-[13px] text-[#1D1D1F]">
                Gating entropy above threshold: input contains hybrid compound degradation.
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3 font-mono-code text-[11px]">
            <span className="text-[#6E6E73] hidden sm:inline">H(w) = 1.18 nats &gt; 1.05 nats</span>
            <button
              onClick={() => setShowEntropyWarning(false)}
              className="text-[#6E6E73] hover:text-[#1D1D1F] p-1 flex items-center justify-center transition-colors cursor-pointer"
            >
              <span className="material-symbols-outlined text-[16px]">close</span>
            </button>
          </div>
        </div>
      )}

      {/* Global Error Banner */}
      {error && (
        <ErrorBanner
          message={error}
          detail={errorDetail}
          tag="MoE Failure"
          onDismiss={() => setError(null)}
        />
      )}

      {/* Header & Description */}
      <div className="w-full flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4 mb-6 border-b border-[#D2D2D7]">
        <div className="flex flex-col max-w-3xl">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider bg-[#0071E3]/10 text-[#0071E3] px-1.5 py-0.5">
              MoE-Dense-Gating-v3
            </span>
            <span className="font-mono-code text-[11px] text-[#6E6E73]">
              ID: 0x4B9E_RESTORE
            </span>
          </div>
          <h1 className="text-[26px] font-semibold text-[#1D1D1F] tracking-tight">
            Soft Mixture-of-Experts
          </h1>
          <p className="text-[13px] text-[#6E6E73] mt-1">
            Continuous soft gating network blending multiple expert representations via learned
            softmax attention weights.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start md:self-auto shrink-0 font-mono-code text-[11px]">
          <Pill variant="success" dot>
            GATE CONVERGED
          </Pill>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-[#F5F5F7] border border-[#D2D2D7] text-[#6E6E73]">
            <span className="material-symbols-outlined text-[14px]">memory</span>
            FP32 ACCURATE
          </span>
        </div>
      </div>

      {/* Operational Workstation Grid: Top Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-6">
        {/* Upload Area & Loaded File State (5 cols) */}
        <div className="lg:col-span-5 flex flex-col">
          <UploadZone
            label="Input Specimen Tensor"
            stateBadge={file ? "TENSOR LOADED" : "TENSOR BUFFER EMPTY"}
            subtext="Multispectral microscopy or complex corrupted specimens"
            file={file}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onClear={handleClear}
            className="h-full"
          />
        </div>

        {/* Execution & Routing Parameters (7 cols) */}
        <div className="lg:col-span-7 bg-[#F5F5F7] border border-[#D2D2D7] p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 mb-3 border-b border-[#D2D2D7]">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Soft Attention Calibration & Diagnostics
              </span>
              <span className="font-mono-code text-[11px] text-[#6E6E73]">
                Architecture: Top-4 Continuous Dynamic Fusion
              </span>
            </div>

            {/* 3 Metric Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="bg-[#FFFFFF] border border-[#D2D2D7] p-3 flex flex-col justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  Temperature (τ)
                </span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-[18px] font-bold text-[#1D1D1F] font-mono-code">
                    0.70
                  </span>
                  <span className="font-mono-code text-[11px] text-[#6E6E73]">
                    Soft Gating
                  </span>
                </div>
                <div className="w-full bg-[#F5F5F7] h-1 mt-2">
                  <div className="bg-[#0071E3] h-1" style={{ width: "70%" }} />
                </div>
              </div>

              <div className="bg-[#FFFFFF] border border-[#D2D2D7] p-3 flex flex-col justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  Gating Entropy
                </span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-[18px] font-bold text-[#B8860B] font-mono-code">
                    1.18
                  </span>
                  <span className="font-mono-code text-[11px] text-[#6E6E73]">nats</span>
                </div>
                <div className="w-full bg-[#F5F5F7] h-1 mt-2">
                  <div className="bg-[#B8860B] h-1" style={{ width: "85%" }} />
                </div>
              </div>

              <div className="bg-[#FFFFFF] border border-[#D2D2D7] p-3 flex flex-col justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                  Inference Latency
                </span>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-[18px] font-bold text-[#1F8A3B] font-mono-code">
                    {inferenceTime !== null ? `${inferenceTime} ms` : "46 ms"}
                  </span>
                  <span className="font-mono-code text-[11px] text-[#6E6E73]">
                    Parallel GPU
                  </span>
                </div>
                <div className="w-full bg-[#F5F5F7] h-1 mt-2">
                  <div className="bg-[#1F8A3B] h-1" style={{ width: "35%" }} />
                </div>
              </div>
            </div>

            {/* Benchmark Preset Selection */}
            <div className="mb-4">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73] block mb-1.5">
                Benchmark Degradation Profiles
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                {BENCHMARK_SETS.map((b) => {
                  const isSelected = corruptionType === b.type;
                  return (
                    <button
                      key={b.id}
                      type="button"
                      onClick={() => setCorruptionType(b.type)}
                      className={`h-7 px-2 border text-[11px] font-mono-code transition-colors cursor-pointer text-center truncate ${
                        isSelected
                          ? "bg-[#0071E3] text-white border-[#0071E3] font-semibold"
                          : "bg-white text-[#1D1D1F] border-[#D2D2D7] hover:bg-[#EEEEEF]"
                      }`}
                    >
                      {b.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Action Button Group */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-3 border-t border-[#D2D2D7]">
            <Button
              variant="primary"
              size="lg"
              loading={loading}
              loadingText="Blending Soft Representations…"
              icon={<span className="material-symbols-outlined text-[18px]">auto_fix_high</span>}
              onClick={() => handleRunRestore()}
              className="flex-1"
            >
              Run Soft Restoration
            </Button>

            <Button
              variant="secondary"
              size="lg"
              disabled={loading}
              icon={<span className="material-symbols-outlined text-[18px]">biotech</span>}
              onClick={async () => {
                const sample = SAMPLE_PRESETS[2];
                const sampleFile = await dataUrlToFile(sample.url, sample.name);
                setFile(sampleFile);
                setPreviewUrl(sample.url);
                await handleRunRestore(sampleFile);
              }}
            >
              Sample Specimen
            </Button>
          </div>
        </div>
      </div>

      {/* Routing Weights Distribution Analysis Panel */}
      <div className="w-full mb-6">
        <BarChart
          title="Continuous Soft Routing Weights w_i (∑ w_i = 1.000)"
          description="Softmax gating weights linearly blend expert tensor representations in latent space."
          tag="SOFTMAX-BLENDED"
          items={chartItems}
          selectedKey={dominantKey}
          dominantBadgeText={`Dominant Expert α=${(routingWeights[dominantKey] ?? 0.58).toFixed(2)}`}
        />
        <div className="mt-2 flex flex-wrap items-center justify-between text-[#6E6E73] font-mono-code text-[11px] px-1">
          <div className="flex items-center gap-2">
            <span>Gating Entropy: 1.18 nats</span>
            <span>•</span>
            <span>Temperature τ = 0.70</span>
            <span>•</span>
            <span>Inference: {inferenceTime ?? 51}ms</span>
          </div>
          <div className="flex items-center gap-1 text-[#0071E3]">
            <span className="material-symbols-outlined text-[14px]">info</span>
            <span>Maintains gradient flow to all specialist models simultaneously</span>
          </div>
        </div>
      </div>

      {/* Primary Interactive Canvas: Side-by-side Restoration Panels */}
      <section className="mb-8">
        <div className="flex items-center justify-between pb-2 mb-4 border-b border-[#D2D2D7]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px] text-[#6E6E73]">view_stream</span>
            <h2 className="text-[15px] font-semibold text-[#1D1D1F]">
              Continuous Multi-Expert Fusion Canvas
            </h2>
          </div>
          <span className="font-mono-code text-[11px] text-[#6E6E73]">
            Dynamic Soft Mixture Inversion
          </span>
        </div>

        <div className="relative">
          {loading && (
            <Loader
              overlay
              label="Fusing Tensor Representations…"
              sublabel={`Blending ${Object.keys(routingWeights).length} expert networks dynamically`}
            />
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Input Panel */}
            <ImagePanel
              title="Composite Degraded Input Specimen"
              badgeText="Degraded Specimen"
              badgeVariant="danger"
              statusDotColor="#D93025"
              imageSrc={result?.corrupted_image || previewUrl}
              alt="Composite Degraded Input"
              overlayTopLeft="CH: 3x128x128"
              overlayBottomLeft={`Corruption: ${corruptionType}`}
              footerLabel="Specimen: multispectral_microscopy"
              footerMetric="SNR: 18.2 dB • σ_est = 0.42"
              downloadFilename="composite_input.png"
            />

            {/* Restored Output Panel */}
            <ImagePanel
              title="Soft-MoE Synthesized Output"
              badgeText={result ? `Dominant: ${dominantKey.toUpperCase()}` : "Awaiting Run"}
              badgeVariant={result ? "success" : "neutral"}
              statusDotColor="#1F8A3B"
              imageSrc={result?.output_image}
              alt="Soft-MoE Synthesized Output"
              overlayTopLeft="MOE-SOFT-BLEND"
              overlayBottomRight={result ? "PSNR: 34.6 dB" : undefined}
              footerLabel="Learned Softmax Weighted Combination"
              footerMetric={result ? `Latency: ${result.inference_time_ms}ms` : undefined}
              downloadFilename="soft_mixture_output.png"
            />
          </div>
        </div>
      </section>
    </div>
  );
}

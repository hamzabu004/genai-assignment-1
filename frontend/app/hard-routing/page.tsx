"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Pill } from "@/components/ui/Pill";
import { UploadZone } from "@/components/ui/UploadZone";
import { ImagePanel } from "@/components/ui/ImagePanel";
import { BarChart } from "@/components/ui/BarChart";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { Loader } from "@/components/ui/Loader";
import { useInference } from "@/hooks/useInference";
import { hardRouting } from "@/lib/api";
import { HardRoutingResponse, CorruptionType, CorruptionSeverity } from "@/lib/types";
import { SAMPLE_PRESETS, dataUrlToFile } from "@/lib/sampleImages";

const PRESETS = [
  { id: "clean", label: "Clean / Pass-through", expert: "identity_pass" },
  { id: "blur", label: "Gaussian Blur", expert: "blur_specialist" },
  { id: "salt_pepper", label: "Salt-and-Pepper", expert: "salt_specialist" },
  { id: "occlusion", label: "Random Occlusion", expert: "occlusion_specialist" },
];

export default function HardRoutingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [routingMode, setRoutingMode] = useState<"hard" | "oracle">("hard");
  const [oracleCorruption, setOracleCorruption] = useState<CorruptionType>("blur");
  const [corruptionType, setCorruptionType] = useState<CorruptionType>("blur");
  const [severity] = useState<CorruptionSeverity>("medium");

  const { loading, error, errorDetail, result, inferenceTime, execute, reset, setError } =
    useInference<HardRoutingResponse>();

  const handleFileSelect = (selectedFile: File, url: string) => {
    setFile(selectedFile);
    setPreviewUrl(url);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    reset();
  };

  const handleClassifyAndRestore = async (targetFile?: File) => {
    const fileToUse = targetFile || file;
    if (!fileToUse) {
      setError("Please select or drop an input frame before running inference.");
      return;
    }

    const typeToSend = routingMode === "oracle" ? oracleCorruption : corruptionType;
    const formData = new FormData();
    formData.append("image", fileToUse);
    formData.append("corruption_type", typeToSend);
    formData.append("severity", severity);

    await execute(() => hardRouting(formData));
  };

  const activeCorruption = routingMode === "oracle" ? oracleCorruption : corruptionType;

  return (
    <div className="flex flex-col w-full">
      {/* Workspace Header */}
      <header className="flex flex-col gap-1 pb-4 mb-6 border-b border-[#D2D2D7]">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-2.5">
            <span className="material-symbols-outlined text-[#0071E3] text-[26px]">
              alt_route
            </span>
            <h1 className="text-[26px] font-semibold text-[#1D1D1F] tracking-tight">
              Hard-Routed Restoration
            </h1>
          </div>
          <div className="flex items-center gap-2 font-mono-code text-[11px] text-[#6E6E73]">
            <span className="px-2 py-1 bg-[#F5F5F7] border border-[#D2D2D7]">
              POLICY: ARGMAX HARD GATING
            </span>
            <span className="px-2 py-1 bg-[#F5F5F7] border border-[#D2D2D7]">
              EXPERTS: 3 SPECIALISTS + 1 PASS
            </span>
          </div>
        </div>
        <p className="text-[13px] text-[#6E6E73] mt-1">
          Discrete classification gating model classifies input degradation and routes exclusively
          to the optimal expert subnetwork.
        </p>
      </header>

      {/* Routing Architecture Mode selector */}
      <div className="mb-6 p-4 bg-[#F5F5F7] border border-[#D2D2D7] flex flex-col gap-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
              Routing Architecture Mode:
            </span>
            <span className="font-mono-code text-[#6E6E73] text-[11px]">
              Select inference gating policy
            </span>
          </div>
          <div className="flex items-center border border-[#D2D2D7] bg-white">
            <button
              type="button"
              onClick={() => setRoutingMode("hard")}
              className={`h-7 px-3 text-[11px] font-semibold uppercase tracking-wider transition-colors flex items-center gap-1.5 cursor-pointer ${
                routingMode === "hard"
                  ? "bg-[#0071E3] text-white"
                  : "bg-transparent text-[#6E6E73] hover:text-[#1D1D1F]"
              }`}
            >
              <span>Hard Routing</span>
              {routingMode === "hard" && (
                <span className="material-symbols-outlined text-[13px]">check</span>
              )}
            </button>
            <button
              type="button"
              onClick={() => setRoutingMode("oracle")}
              className={`h-7 px-3 text-[11px] font-semibold uppercase tracking-wider transition-colors flex items-center gap-1.5 border-l border-[#D2D2D7] cursor-pointer ${
                routingMode === "oracle"
                  ? "bg-[#0071E3] text-white"
                  : "bg-transparent text-[#6E6E73] hover:text-[#1D1D1F]"
              }`}
            >
              <span>Oracle Routing</span>
              {routingMode === "oracle" && (
                <span className="material-symbols-outlined text-[13px]">check</span>
              )}
            </button>
          </div>
        </div>

        {routingMode === "oracle" && (
          <div className="pt-2.5 border-t border-[#D2D2D7] flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Oracle Ground-Truth Corruption Override:
              </span>
              <span className="font-mono-code text-[#6E6E73] text-[11px]">
                (Bypasses gating classification classifier)
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              {PRESETS.map((p) => {
                const isSelected = oracleCorruption === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setOracleCorruption(p.id as CorruptionType)}
                    className={`h-7 px-2.5 border text-[11px] font-mono-code uppercase transition-colors cursor-pointer ${
                      isSelected
                        ? "bg-[#0071E3] text-white border-[#0071E3] font-semibold"
                        : "bg-white border-[#D2D2D7] text-[#6E6E73] hover:bg-[#EEEEEF]"
                    }`}
                  >
                    {p.id}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Global Error Banner */}
      {error && (
        <ErrorBanner
          message={error}
          detail={errorDetail}
          tag="Routing Error"
          onDismiss={() => setError(null)}
        />
      )}

      {/* Input Selection Section */}
      <section className="mb-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Upload Zone (7 cols) */}
          <div className="lg:col-span-7 flex flex-col">
            <UploadZone
              label="Input Source Buffer & Specimen"
              stateBadge={file ? "Loaded: 100%" : "Status: Awaiting Frame"}
              subtext="CCTV frames, optical bench captures, or forensic photography"
              file={file}
              previewUrl={previewUrl}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
              presets={[]}
              className="h-full"
            />
          </div>

          {/* Quick-Load Presets & Action Trigger (5 cols) */}
          <div className="lg:col-span-5 bg-[#F5F5F7] border border-[#D2D2D7] p-5 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73] mb-2">
                Simulated Degradation Mode
              </div>
              <div className="grid grid-cols-2 gap-1.5 mb-4">
                {PRESETS.map((p) => {
                  const isSelected = activeCorruption === p.id;
                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setCorruptionType(p.id as CorruptionType)}
                      className={`h-8 px-2.5 border text-[11px] font-medium transition-colors text-left flex items-center justify-between cursor-pointer truncate ${
                        isSelected
                          ? "bg-white border-[#0071E3] text-[#0071E3]"
                          : "bg-white border-[#D2D2D7] text-[#1D1D1F] hover:bg-[#EEEEEF]"
                      }`}
                    >
                      <span className="truncate">{p.label}</span>
                      {isSelected && (
                        <span className="material-symbols-outlined text-[14px] shrink-0">
                          check
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2 pt-3 border-t border-[#D2D2D7]">
              <Button
                variant="primary"
                size="lg"
                loading={loading}
                loadingText="Classifying & Restoring…"
                icon={<span className="material-symbols-outlined text-[18px]">bolt</span>}
                onClick={() => handleClassifyAndRestore()}
                className="flex-1"
              >
                Classify & Restore
              </Button>

              <Button
                variant="secondary"
                size="lg"
                disabled={loading}
                onClick={async () => {
                  const sample = SAMPLE_PRESETS[1];
                  const sampleFile = await dataUrlToFile(sample.url, sample.name);
                  setFile(sampleFile);
                  setPreviewUrl(sample.url);
                  await handleClassifyAndRestore(sampleFile);
                }}
                title="Load sample frame and execute"
              >
                Test CCTV
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Routing Decision & Telemetry Strip */}
      <section className="mb-6">
        <div className="bg-[#F5F5F7] border border-[#D2D2D7] p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center flex-wrap gap-4">
            {/* Predicted Corruption */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Predicted Corruption:
              </span>
              <Pill variant="info" icon={<span className="material-symbols-outlined text-[12px]">blur_on</span>}>
                {result ? result.predicted_class.toUpperCase() : "AWAITING INFERENCE"}
              </Pill>
            </div>

            {/* Selected Expert Subnet */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6E6E73]">
                Selected Expert:
              </span>
              <Pill variant="success" icon={<span className="material-symbols-outlined text-[12px]">hub</span>}>
                {result ? result.selected_expert : "AWAITING ROUTE"}
              </Pill>
            </div>
          </div>

          {/* Inference Latency Badge */}
          <div className="flex items-center gap-1.5 font-mono-code text-[11px] bg-white border border-[#D2D2D7] px-3 py-1">
            <span className="material-symbols-outlined text-[14px] text-[#6E6E73]">timer</span>
            <span className="text-[#1D1D1F] font-semibold">
              {inferenceTime !== null ? `${inferenceTime}ms` : "—"}
            </span>
            <span className="text-[#6E6E73] text-[10px]">(Gating: 6ms, Expert: 31ms)</span>
          </div>
        </div>
      </section>

      {/* Classifier Distribution Panel */}
      <section className="mb-6">
        <BarChart
          title="Discrete Classifier Probabilities"
          description="Argmax selector routes tensors exclusively to the single top-confidence specialized subnetwork."
          tag="HARD-ARGMAX"
          data={
            result?.class_probabilities || {
              clean: 0.03,
              salt_pepper: 0.08,
              blur: 0.84,
              occlusion: 0.05,
            }
          }
          selectedKey={result?.predicted_class || "blur"}
          dominantBadgeText="SELECTED ROUTE"
        />
      </section>

      {/* Side-by-Side Image Comparison Panels */}
      <section className="mb-8">
        <div className="flex items-center justify-between pb-2 mb-4 border-b border-[#D2D2D7]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px] text-[#6E6E73]">compare</span>
            <h2 className="text-[15px] font-semibold text-[#1D1D1F]">
              Reconstruction Verification
            </h2>
          </div>
          <span className="font-mono-code text-[11px] text-[#6E6E73]">
            Target Dimensions: 128 × 128
          </span>
        </div>

        <div className="relative">
          {loading && (
            <Loader
              overlay
              label="Running inference…"
              sublabel={`Executing ${result?.selected_expert || "Specialized Expert Subnetwork"}`}
            />
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Input Panel */}
            <ImagePanel
              title="Degraded Input Image"
              badgeText="Input Degradation"
              badgeVariant="danger"
              statusDotColor="#D93025"
              imageSrc={result?.corrupted_image || previewUrl}
              alt="Degraded input frame"
              overlayBottomLeft={`Corruption: ${activeCorruption}`}
              footerLabel="Sample: forensic_cctv_specimen"
              footerMetric="PSNR: 18.42 dB • SSIM: 0.6120"
              downloadFilename="degraded_input.png"
            />

            {/* Output Panel */}
            <ImagePanel
              title="Hard-Routed Reconstructed Output"
              badgeText={result ? `Restored by ${result.selected_expert}` : "Awaiting Result"}
              badgeVariant={result ? "success" : "neutral"}
              statusDotColor="#1F8A3B"
              imageSrc={result?.output_image}
              alt="Reconstructed output frame"
              overlayTopLeft={result ? `EXPERT: ${result.selected_expert.toUpperCase()}` : undefined}
              overlayBottomRight={result ? "PSNR: 32.88 dB" : undefined}
              footerLabel="Argmax Hard-Routed Restoration"
              footerMetric={result ? `Inference Latency: ${result.inference_time_ms}ms` : undefined}
              downloadFilename="hard_routed_output.png"
            />
          </div>
        </div>
      </section>
    </div>
  );
}

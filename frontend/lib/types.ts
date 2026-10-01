// Mirrors backend Pydantic schemas — see Plan 2, Section 3

export type CorruptionType = "clean" | "salt_pepper" | "blur" | "occlusion";
export type CorruptionSeverity = "low" | "medium" | "high";
export type SketchStyle = "style_1" | "style_2" | "style_3";

export interface UniversalRestorationResponse {
  input_image: string;
  corrupted_image: string;
  output_image: string;
  error_map_image: string;
  corruption_applied: {
    type: string;
    severity: string;
    params: Record<string, unknown>;
  };
  inference_time_ms: number;
  quality_metrics: {
    psnr_db: number;
    ssim: number;
  };
  sample_filename: string | null;
}

export interface ValidationSample {
  filename: string;
  corruption_type: CorruptionType;
  severity: string;
  params: Record<string, unknown>;
}

export interface HardRoutingResponse {
  input_image: string;
  corrupted_image: string;
  output_image: string;
  class_probabilities: Record<string, number>;
  predicted_class: string;
  selected_expert: string;
  inference_time_ms: number;
}

export interface SoftMixtureResponse {
  input_image: string;
  corrupted_image: string;
  output_image: string;
  routing_weights: Record<string, number>;
  dominant_expert: string;
  inference_time_ms: number;
}

export interface FaceToSketchResponse {
  input_image: string;
  sketch_image: string;
  style_used: string;
  inference_time_ms: number;
}

export interface ApiErrorResponse {
  error: boolean;
  message: string;
  detail?: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
}

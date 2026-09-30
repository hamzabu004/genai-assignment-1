import {
  UniversalRestorationResponse,
  HardRoutingResponse,
  SoftMixtureResponse,
  FaceToSketchResponse,
  HealthResponse,
} from "./types";
import { createGeometricFaceSvg } from "./sampleImages";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  detail?: string;
  constructor(message: string, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
  }
}

async function postForm<T>(path: string, form: FormData, fallbackGenerator?: () => T): Promise<T> {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      let errorMessage = `Request failed with status ${res.status}`;
      let detail: string | undefined;
      try {
        const errorJson = await res.json();
        if (errorJson.message) errorMessage = errorJson.message;
        if (errorJson.detail) detail = errorJson.detail;
      } catch {
        // use default message
      }
      throw new ApiError(errorMessage, detail);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    // If backend is unreachable (Failed to fetch) and fallbackGenerator is provided, use fallback for interactive demo
    const isNetworkError = err instanceof TypeError && err.message.includes("fetch");
    if (isNetworkError && fallbackGenerator) {
      console.warn(`[API] Backend unreachable at ${BASE_URL}${path}. Using simulated response.`);
      await new Promise((resolve) => setTimeout(resolve, 800)); // simulate latency
      return fallbackGenerator();
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(err instanceof Error ? err.message : "Network error during inference");
  }
}

export async function universalRestoration(form: FormData): Promise<UniversalRestorationResponse> {
  return postForm<UniversalRestorationResponse>("/universal-restoration", form, () => {
    const corruptionType = (form.get("corruption_type") as string) || "salt_pepper";
    const severity = (form.get("severity") as string) || "medium";
    const sample = createGeometricFaceSvg(1);
    return {
      input_image: sample,
      corrupted_image: sample,
      output_image: sample,
      error_map_image: sample,
      corruption_applied: {
        type: corruptionType,
        severity,
        params: {
          intensity: severity === "high" ? 0.75 : severity === "low" ? 0.25 : 0.5,
          sigma: 1.5,
        },
      },
      inference_time_ms: 42.3,
    };
  });
}

export async function hardRouting(form: FormData): Promise<HardRoutingResponse> {
  return postForm<HardRoutingResponse>("/hard-routing", form, () => {
    const corruptionType = (form.get("corruption_type") as string) || "blur";
    const sample = createGeometricFaceSvg(2);
    const predictedClass = corruptionType === "clean" ? "clean" : corruptionType;
    const expertName = `${predictedClass}_specialist`;
    return {
      input_image: sample,
      corrupted_image: sample,
      output_image: sample,
      class_probabilities: {
        clean: predictedClass === "clean" ? 0.88 : 0.04,
        salt_pepper: predictedClass === "salt_pepper" ? 0.84 : 0.06,
        blur: predictedClass === "blur" ? 0.86 : 0.05,
        occlusion: predictedClass === "occlusion" ? 0.82 : 0.05,
      },
      predicted_class: predictedClass,
      selected_expert: expertName,
      inference_time_ms: 37.1,
    };
  });
}

export async function softMixture(form: FormData): Promise<SoftMixtureResponse> {
  return postForm<SoftMixtureResponse>("/soft-mixture", form, () => {
    const corruptionType = (form.get("corruption_type") as string) || "blur";
    const dominant = corruptionType === "clean" ? "identity" : corruptionType;
    const sample = createGeometricFaceSvg(3);
    return {
      input_image: sample,
      corrupted_image: sample,
      output_image: sample,
      routing_weights: {
        identity: dominant === "identity" ? 0.72 : 0.08,
        salt_pepper: dominant === "salt_pepper" ? 0.68 : 0.1,
        blur: dominant === "blur" ? 0.76 : 0.07,
        occlusion: dominant === "occlusion" ? 0.74 : 0.09,
      },
      dominant_expert: dominant,
      inference_time_ms: 51.4,
    };
  });
}

export async function faceToSketch(form: FormData): Promise<FaceToSketchResponse> {
  return postForm<FaceToSketchResponse>("/face-to-sketch", form, () => {
    const style = (form.get("style") as string) || "style_2";
    const sample = createGeometricFaceSvg(1);
    return {
      input_image: sample,
      sketch_image: sample,
      style_used: style,
      inference_time_ms: 63.0,
    };
  });
}

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error("Health check failed");
    return await res.json();
  } catch {
    return {
      status: "offline",
      models_loaded: [],
    };
  }
}

"use client";

import { useState, useCallback } from "react";
import { ApiError } from "@/lib/api";

export interface UseInferenceReturn<T> {
  loading: boolean;
  error: string | null;
  errorDetail: string | null;
  result: T | null;
  inferenceTime: number | null;
  execute: (apiCall: () => Promise<T>) => Promise<T | null>;
  reset: () => void;
  setError: (err: string | null, detail?: string | null) => void;
  setResult: React.Dispatch<React.SetStateAction<T | null>>;
}

export function useInference<T = unknown>(): UseInferenceReturn<T> {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setErrorState] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [result, setResult] = useState<T | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);

  const setError = useCallback((err: string | null, detail: string | null = null) => {
    setErrorState(err);
    setErrorDetail(detail);
  }, []);

  const reset = useCallback(() => {
    setLoading(false);
    setErrorState(null);
    setErrorDetail(null);
    setResult(null);
    setInferenceTime(null);
  }, []);

  const execute = useCallback(
    async (apiCall: () => Promise<T>): Promise<T | null> => {
      setLoading(true);
      setErrorState(null);
      setErrorDetail(null);
      const startTime = performance.now();

      try {
        const data = await apiCall();
        const duration = performance.now() - startTime;
        setResult(data);
        // If data contains inference_time_ms, use it; otherwise use client duration
        if (data && typeof data === "object" && "inference_time_ms" in data) {
          setInferenceTime((data as { inference_time_ms: number }).inference_time_ms);
        } else {
          setInferenceTime(Math.round(duration));
        }
        return data;
      } catch (err: unknown) {
        if (err instanceof ApiError) {
          setErrorState(err.message);
          setErrorDetail(err.detail || null);
        } else if (err instanceof Error) {
          setErrorState(err.message);
        } else {
          setErrorState("An unexpected error occurred during inference.");
        }
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    errorDetail,
    result,
    inferenceTime,
    execute,
    reset,
    setError,
    setResult,
  };
}

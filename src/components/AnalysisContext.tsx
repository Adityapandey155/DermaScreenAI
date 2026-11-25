"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { PredictionResult } from "@/ml/types";

/**
 * Keeps the selected photo and the latest result in MEMORY only.
 * Nothing is written to localStorage, cookies or any server: closing or reloading the tab discards everything.
 */
interface AnalysisState {
  file: File | null;
  previewUrl: string | null;
  result: PredictionResult | null;
  setFile: (file: File | null) => void;
  setResult: (r: PredictionResult | null) => void;
  reset: () => void;
}

const Ctx = createContext<AnalysisState | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [file, setFileState] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const urlRef = useRef<string | null>(null);

  const setFile = useCallback((f: File | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = f ? URL.createObjectURL(f) : null;
    setFileState(f);
    setPreviewUrl(urlRef.current);
    setResult(null);
  }, []);

  const reset = useCallback(() => setFile(null), [setFile]);

  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  const value = useMemo(
    () => ({ file, previewUrl, result, setFile, setResult, reset }),
    [file, previewUrl, result, setFile, reset],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAnalysis() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAnalysis must be used inside AnalysisProvider");
  return v;
}

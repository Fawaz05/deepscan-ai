"use client";

import { useEffect, useState } from "react";

const STEPS = [
  "Preprocessing image...",
  "Running MobileNetV2 inference...",
  "Computing Grad-CAM++ heatmaps...",
  "Performing forensic FFT analysis...",
  "Generating AI forensic explanation...",
];

export function LoadingSpinner() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % STEPS.length), 900);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center py-16 animate-fade-in">
      <div className="h-16 w-16 animate-spin rounded-full border-4 border-brand-500/30 border-t-brand-500" />
      <p className="mt-6 font-semibold text-brand-500">Analyzing with deep neural forensics...</p>
      <p className="mt-2 text-sm text-[var(--muted)]">{STEPS[step]}</p>
    </div>
  );
}

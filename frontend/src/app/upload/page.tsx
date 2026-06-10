"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import { uploadImage, type AnalysisReport } from "@/lib/api";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ResultsView } from "@/components/ResultsView";

export default function UploadPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await uploadImage(files[0]);
      setReport(result);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".gif"] },
    maxFiles: 1,
    disabled: loading,
  });

  return (
    <div>
      <h1 className="mb-2 text-3xl font-bold">Upload Image</h1>
      <p className="mb-8 text-[var(--muted)]">Drop an image to run the full forensic analysis pipeline.</p>

      {!report && !loading && (
        <div
          {...getRootProps()}
          className={`card cursor-pointer border-2 border-dashed text-center transition ${
            isDragActive ? "border-brand-500 bg-brand-500/5" : "border-[var(--border)] hover:border-brand-500/50"
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto mb-4 h-12 w-12 text-brand-500" />
          <p className="font-medium">{isDragActive ? "Drop image here..." : "Drag & drop an image, or click to browse"}</p>
          <p className="mt-2 text-sm text-[var(--muted)]">JPEG, PNG, WEBP — max 20 MB</p>
        </div>
      )}

      {loading && <LoadingSpinner />}
      {error && <div className="card border-red-500/30 bg-red-500/10 text-red-500">{error}</div>}
      {report && <ResultsView report={report} />}
    </div>
  );
}

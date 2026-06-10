"use client";

import { useState } from "react";
import { Link2 } from "lucide-react";
import { analyzeUrl, type AnalysisReport } from "@/lib/api";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ResultsView } from "@/components/ResultsView";

export default function AnalyzePage() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await analyzeUrl(url.trim());
      setReport(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-2 text-3xl font-bold">Analyze URL</h1>
      <p className="mb-8 text-[var(--muted)]">Enter a direct image URL for forensic deepfake analysis.</p>

      {!report && !loading && (
        <form onSubmit={handleSubmit} className="card">
          <label className="mb-2 block text-sm font-medium">Image URL</label>
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Link2 className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--muted)]" />
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/image.jpg"
                className="w-full rounded-lg border border-[var(--border)] bg-transparent py-2.5 pl-10 pr-4 text-sm focus:border-brand-500 focus:outline-none"
                required
              />
            </div>
            <button type="submit" className="btn-primary">Analyze</button>
          </div>
        </form>
      )}

      {loading && <LoadingSpinner />}
      {error && <div className="card border-red-500/30 bg-red-500/10 text-red-500">{error}</div>}
      {report && <ResultsView report={report} />}
    </div>
  );
}

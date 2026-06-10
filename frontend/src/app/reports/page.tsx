"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listReports } from "@/lib/api";

interface ReportSummary {
  report_id: string;
  prediction: string;
  confidence: number;
  trust_score: number;
  risk_level: string;
  timestamp: string;
}

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listReports(50)
      .then((data) => setReports(data.reports || []))
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="mb-2 text-3xl font-bold">Reports</h1>
      <p className="mb-8 text-[var(--muted)]">Recent forensic analysis reports.</p>

      {loading && <p className="text-[var(--muted)]">Loading reports...</p>}

      {!loading && reports.length === 0 && (
        <div className="card text-center text-[var(--muted)]">No reports yet. Upload an image to get started.</div>
      )}

      <div className="space-y-3">
        {reports.map((r) => (
          <Link
            key={r.report_id}
            href={`/results/${r.report_id}`}
            className="card flex flex-wrap items-center justify-between gap-4 transition hover:border-brand-500/50"
          >
            <div>
              <p className="font-medium">{r.report_id}</p>
              <p className="text-xs text-[var(--muted)]">{r.timestamp}</p>
            </div>
            <div className="flex gap-6 text-sm">
              <span className={r.prediction === "Fake" ? "text-red-500" : "text-green-500"}>{r.prediction}</span>
              <span>{r.confidence}%</span>
              <span>Trust: {r.trust_score}</span>
              <span>{r.risk_level} risk</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

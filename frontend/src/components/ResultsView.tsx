"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Download, ShieldAlert } from "lucide-react";
import type { AnalysisReport } from "@/lib/api";
import { downloadReportUrl } from "@/lib/api";

function riskColor(risk: string) {
  if (risk === "High") return "text-red-500";
  if (risk === "Medium") return "text-amber-500";
  return "text-green-500";
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mb-3">
      <div className="mb-1 flex justify-between text-xs text-[var(--muted)]">
        <span>{label}</span>
        <span className="font-bold" style={{ color }}>{value}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-gray-800/30">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, value)}%` }}
          transition={{ duration: 1 }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
    </div>
  );
}

export function ResultsView({ report }: { report: AnalysisReport }) {
  const [heatmapOpacity, setHeatmapOpacity] = useState(50);
  const predColor = report.prediction === "Fake" ? "#ef4444" : "#22c55e";

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold">Analysis Results</h2>
          <p className="text-sm text-[var(--muted)]">Report ID: {report.report_id}</p>
        </div>
        <a
          href={downloadReportUrl(report.report_id)}
          className="btn-primary gap-2"
          download
        >
          <Download className="h-4 w-4" /> Download Report
        </a>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="card text-center">
          <p className="text-sm text-[var(--muted)]">Prediction</p>
          <p className="text-3xl font-bold" style={{ color: predColor }}>{report.prediction}</p>
          <p className="text-sm">{report.confidence}% confidence</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-[var(--muted)]">Trust Score</p>
          <p className="text-3xl font-bold">{report.trust_score}</p>
          <p className="text-sm">out of 100</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-[var(--muted)]">Risk Level</p>
          <p className={`text-3xl font-bold ${riskColor(report.risk_level)}`}>{report.risk_level}</p>
        </div>
      </div>

      {report.misuse_warning && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-500 shrink-0" />
          <p className="text-sm">{report.misuse_warning}</p>
        </div>
      )}

      <div className="card">
        <h3 className="mb-3 font-semibold">Forensic Explanation</h3>
        <p className="leading-relaxed text-[var(--muted)]">{report.forensic_explanation}</p>
      </div>

      {report.forensic_scores && (
        <div className="card">
          <h3 className="mb-4 font-semibold">Forensic Scores</h3>
          <ScoreBar label="Authenticity" value={report.forensic_scores.authenticity_score} color="#22c55e" />
          <ScoreBar label="Suspicion" value={report.forensic_scores.suspicion_score} color="#ef4444" />
          <ScoreBar label="Forensic Confidence" value={report.forensic_scores.forensic_confidence} color="#6366f1" />
          {report.forensic_scores.manipulation_indicators.length > 0 && (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-[var(--muted)]">
              {report.forensic_scores.manipulation_indicators.map((ind, i) => (
                <li key={i}>{ind}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {report.social_safety && (
        <div className="card">
          <div className="mb-4 flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-brand-500" />
            <h3 className="font-semibold">Social Media Safety — {report.social_safety.safety_category}</h3>
          </div>
          <ScoreBar label="Cyberbullying Risk" value={report.social_safety.cyberbullying_risk} color="#ef4444" />
          <ScoreBar label="Misinformation Risk" value={report.social_safety.misinformation_risk} color="#f59e0b" />
          <ScoreBar label="Impersonation Risk" value={report.social_safety.impersonation_risk} color="#8b5cf6" />
          <ScoreBar label="Identity Abuse Risk" value={report.social_safety.identity_abuse_risk} color="#ec4899" />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {report.annotated_image && (
          <div className="card">
            <h3 className="mb-3 font-semibold">Suspicious Regions</h3>
            <img src={`data:image/png;base64,${report.annotated_image}`} alt="Annotated" className="w-full rounded-lg" />
          </div>
        )}
        {report.heatmap_image && report.annotated_image && (
          <div className="card">
            <h3 className="mb-3 font-semibold">Grad-CAM++ Heatmap</h3>
            <div className="relative overflow-hidden rounded-lg">
              <img src={`data:image/png;base64,${report.annotated_image}`} alt="Base" className="w-full" />
              <img
                src={`data:image/png;base64,${report.heatmap_image}`}
                alt="Heatmap"
                className="absolute inset-0 w-full"
                style={{ opacity: heatmapOpacity / 100 }}
              />
            </div>
            <input
              type="range" min={0} max={100} value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="mt-3 w-full accent-brand-500"
            />
          </div>
        )}
        {report.fft_image && (
          <div className="card">
            <h3 className="mb-3 font-semibold">FFT Frequency Spectrum</h3>
            <img src={`data:image/png;base64,${report.fft_image}`} alt="FFT" className="w-full rounded-lg" />
            <p className="mt-2 text-xs text-[var(--muted)]">
              High-freq energy: {report.fft_analysis?.high_freq_energy} | Anomaly: {report.fft_analysis?.fft_anomaly_score}/100
            </p>
          </div>
        )}
      </div>

      {report.suspicious_regions?.length > 0 && (
        <div className="card">
          <h3 className="mb-4 font-semibold">Region Analysis</h3>
          <div className="space-y-3">
            {report.suspicious_regions.map((r) => (
              <div key={r.index} className="rounded-lg border border-[var(--border)] p-4">
                <div className="mb-1 flex justify-between">
                  <span className="font-medium">Region {r.index}</span>
                  <span className="text-sm text-brand-500">{Math.round(r.intensity * 100)}% intensity</span>
                </div>
                <p className="text-sm text-[var(--muted)]">{r.caption}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

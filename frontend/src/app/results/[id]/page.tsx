"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getReport } from "@/lib/api";
import type { AnalysisReport } from "@/lib/api";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ResultsView } from "@/components/ResultsView";

export default function ResultPage() {
  const params = useParams();
  const id = params.id as string;
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReport(id)
      .then(setReport)
      .catch(() => setError("Report not found"));
  }, [id]);

  if (error) return <div className="card text-red-500">{error}</div>;
  if (!report) return <LoadingSpinner />;
  return <ResultsView report={report} />;
}

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API = `${API_BASE}/api/v1`;

export interface AnalysisReport {
  report_id: string;
  prediction: string;
  confidence: number;
  trust_score: number;
  risk_level: string;
  forensic_explanation: string;
  misuse_warning?: string | null;
  heatmap_image?: string | null;
  annotated_image?: string | null;
  fft_image?: string | null;
  suspicious_regions: Array<{
    index: number;
    bbox: number[];
    intensity: number;
    caption: string;
  }>;
  region_captions: string[];
  fft_analysis: Record<string, number>;
  score_components: Record<string, number>;
  forensic_scores?: {
    authenticity_score: number;
    suspicion_score: number;
    forensic_confidence: number;
    manipulation_indicators: string[];
  };
  social_safety?: {
    safety_category: string;
    cyberbullying_risk: number;
    misinformation_risk: number;
    impersonation_risk: number;
    identity_abuse_risk: number;
    warnings: string[];
  };
  texture_analysis?: Record<string, unknown>;
  noise_analysis?: Record<string, unknown>;
  compression_analysis?: Record<string, unknown>;
  timestamp: string;
}

export async function uploadImage(file: File): Promise<AnalysisReport> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axios.post<AnalysisReport>(`${API}/upload-image`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120000,
  });
  return data;
}

export async function analyzeUrl(url: string): Promise<AnalysisReport> {
  const { data } = await axios.post<AnalysisReport>(`${API}/analyze-url`, { url }, { timeout: 120000 });
  return data;
}

export async function getReport(reportId: string) {
  const { data } = await axios.get(`${API}/report/${reportId}`);
  return data;
}

export async function listReports(limit = 20) {
  const { data } = await axios.get(`${API}/reports`, { params: { limit } });
  return data;
}

export function downloadReportUrl(reportId: string) {
  return `${API}/download-report/${reportId}`;
}

export async function healthCheck() {
  const { data } = await axios.get(`${API}/health`);
  return data;
}

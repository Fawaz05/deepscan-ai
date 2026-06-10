// src/App.js — Deepfake Detection Platform UI
import React, { useState, useCallback, useRef } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─────────────────────────────────────────────
// Color / style utilities
// ─────────────────────────────────────────────
const getRiskColor = (risk) => ({
  High:   "#ef4444",
  Medium: "#f59e0b",
  Low:    "#22c55e",
}[risk] || "#6b7280");

const getPredColor = (pred) =>
  pred === "Fake" ? "#ef4444" : "#22c55e";

const getTrustColor = (score) => {
  if (score < 30) return "#ef4444";
  if (score < 60) return "#f59e0b";
  return "#22c55e";
};

// ─────────────────────────────────────────────
// Animated Progress Bar
// ─────────────────────────────────────────────
function AnimatedBar({ value, color, label }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: "#9ca3af" }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color }}>{value}%</span>
      </div>
      <div style={{ height: 8, background: "#1f2937", borderRadius: 4, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${value}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            borderRadius: 4,
            transition: "width 1s ease-out",
          }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Region Card
// ─────────────────────────────────────────────
function RegionCard({ region, index }) {
  const intensity = Math.round(region.intensity * 100);
  return (
    <div style={{
      background: "#111827",
      border: "1px solid #374151",
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontWeight: 700, color: "#f3f4f6" }}>Region {index + 1}</span>
        <span style={{
          padding: "2px 10px",
          borderRadius: 999,
          background: intensity > 70 ? "#ef444422" : intensity > 50 ? "#f59e0b22" : "#22c55e22",
          color: intensity > 70 ? "#ef4444" : intensity > 50 ? "#f59e0b" : "#22c55e",
          fontSize: 12,
          fontWeight: 700,
        }}>
          {intensity}% intensity
        </span>
      </div>
      <p style={{ fontSize: 13, color: "#9ca3af", lineHeight: 1.6, margin: 0 }}>
        {region.caption || "Visual inconsistencies detected in this region."}
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────
// Heatmap Overlay Slider
// ─────────────────────────────────────────────
function HeatmapSlider({ original, heatmap }) {
  const [opacity, setOpacity] = useState(50);
  return (
    <div>
      <div style={{ position: "relative", borderRadius: 12, overflow: "hidden" }}>
        <img src={`data:image/png;base64,${original}`} alt="Original"
          style={{ width: "100%", display: "block" }} />
        <img src={`data:image/png;base64,${heatmap}`} alt="Heatmap"
          style={{
            position: "absolute", top: 0, left: 0, width: "100%",
            opacity: opacity / 100, transition: "opacity 0.2s",
          }} />
      </div>
      <div style={{ marginTop: 12 }}>
        <label style={{ fontSize: 12, color: "#9ca3af", display: "flex", justifyContent: "space-between" }}>
          <span>Heatmap Overlay</span>
          <span>{opacity}%</span>
        </label>
        <input type="range" min="0" max="100" value={opacity}
          onChange={e => setOpacity(Number(e.target.value))}
          style={{ width: "100%", accentColor: "#6366f1", marginTop: 4 }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Loading Animation
// ─────────────────────────────────────────────
function LoadingScreen() {
  const steps = [
    "Preprocessing image...",
    "Running MobileNetV2 inference...",
    "Computing Grad-CAM++ heatmaps...",
    "Extracting suspicious regions...",
    "Performing FFT frequency analysis...",
    "Generating forensic explanation...",
  ];
  const [step, setStep] = React.useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setStep(s => (s + 1) % steps.length);
    }, 900);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "60px 20px" }}>
      <div style={{
        width: 80, height: 80, margin: "0 auto 24px",
        border: "3px solid #374151",
        borderTopColor: "#6366f1",
        borderRadius: "50%",
        animation: "spin 1s linear infinite",
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <p style={{ color: "#6366f1", fontWeight: 600, fontSize: 16, margin: "0 0 8px" }}>
        Analyzing image using deep neural forensics...
      </p>
      <p style={{ color: "#6b7280", fontSize: 14 }}>{steps[step]}</p>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────
export default function App() {
  const [mode, setMode] = useState("upload");   // "upload" | "url"
  const [url, setUrl] = useState("");
  const [previewSrc, setPreviewSrc] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("heatmap");

  // ── Dropzone ──
  const onDrop = useCallback(accepted => {
    if (!accepted.length) return;
    const file = accepted[0];
    setPreviewFile(file);
    setPreviewSrc(URL.createObjectURL(file));
    setReport(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    maxFiles: 1,
  });

  // ── Submit ──
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      let res;
      if (mode === "upload" && previewFile) {
        const form = new FormData();
        form.append("file", previewFile);
        res = await axios.post(`${API}/upload-image`, form);
      } else if (mode === "url" && url) {
        res = await axios.post(`${API}/analyze-url`, { url });
      } else {
        setError("Please select an image or enter a URL.");
        setLoading(false);
        return;
      }
      setReport(res.data);
      setActiveTab("heatmap");
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        "Analysis failed. Please check the server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setReport(null);
    setPreviewSrc(null);
    setPreviewFile(null);
    setUrl("");
    setError(null);
  };

  // ──────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%)",
      color: "#f3f4f6",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>

      {/* Header */}
      <header style={{
        borderBottom: "1px solid #1f2937",
        padding: "20px 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(10,10,20,0.8)",
        backdropFilter: "blur(12px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18,
          }}>🔍</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 18, letterSpacing: "-0.5px" }}>
              DeepForensics
            </div>
            <div style={{ fontSize: 11, color: "#6b7280" }}>AI-Powered Deepfake Detection</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["upload", "url"].map(m => (
            <button key={m} onClick={() => { setMode(m); reset(); }}
              style={{
                padding: "8px 16px",
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 13,
                background: mode === m
                  ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                  : "transparent",
                color: mode === m ? "#fff" : "#6b7280",
                transition: "all 0.2s",
              }}>
              {m === "upload" ? "📤 Upload" : "🔗 URL"}
            </button>
          ))}
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 24px" }}>

        {/* Hero */}
        {!report && !loading && (
          <div style={{ textAlign: "center", marginBottom: 40 }}>
            <h1 style={{
              fontSize: 42,
              fontWeight: 900,
              letterSpacing: "-1.5px",
              margin: "0 0 12px",
              background: "linear-gradient(135deg, #f3f4f6 0%, #a5b4fc 60%, #8b5cf6 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}>
              Forensic Deepfake Analysis
            </h1>
            <p style={{ color: "#6b7280", fontSize: 16, maxWidth: 520, margin: "0 auto" }}>
              Upload an image or paste a URL. Our XAI pipeline provides Grad-CAM++ heatmaps,
              region analysis, frequency forensics, and a forensic-grade explanation.
            </p>
          </div>
        )}

        {/* Input Section */}
        {!report && !loading && (
          <div style={{
            background: "#0d1117",
            border: "1px solid #21262d",
            borderRadius: 20,
            padding: 32,
            maxWidth: 700,
            margin: "0 auto 40px",
          }}>
            {mode === "upload" ? (
              <div>
                <div {...getRootProps()} style={{
                  border: `2px dashed ${isDragActive ? "#6366f1" : "#374151"}`,
                  borderRadius: 16,
                  padding: "40px 20px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: isDragActive ? "#6366f110" : "transparent",
                  transition: "all 0.2s",
                }}>
                  <input {...getInputProps()} />
                  <div style={{ fontSize: 40, marginBottom: 12 }}>
                    {previewSrc ? "🖼️" : "📁"}
                  </div>
                  {previewSrc ? (
                    <div>
                      <img src={previewSrc} alt="Preview" style={{
                        maxHeight: 200, maxWidth: "100%",
                        borderRadius: 12, marginBottom: 12,
                        objectFit: "contain",
                      }} />
                      <p style={{ color: "#9ca3af", fontSize: 14, margin: 0 }}>
                        Click or drag to change image
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p style={{ color: "#9ca3af", margin: "0 0 4px", fontSize: 15 }}>
                        {isDragActive ? "Drop image here..." : "Drag & drop an image here"}
                      </p>
                      <p style={{ color: "#4b5563", margin: 0, fontSize: 13 }}>
                        JPEG, PNG, WEBP supported
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div>
                <label style={{ display: "block", marginBottom: 8, fontSize: 14, color: "#9ca3af" }}>
                  Image URL
                </label>
                <input
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                  style={{
                    width: "100%", padding: "12px 16px",
                    background: "#161b22", border: "1px solid #30363d",
                    borderRadius: 10, color: "#f3f4f6",
                    fontSize: 14, outline: "none",
                    boxSizing: "border-box",
                  }}
                />
                <p style={{ fontSize: 12, color: "#4b5563", marginTop: 8 }}>
                  Supports direct image links. Instagram public links may not work due to CORS restrictions.
                </p>
              </div>
            )}

            {error && (
              <div style={{
                marginTop: 16, padding: 12,
                background: "#ef444420", border: "1px solid #ef444440",
                borderRadius: 10, color: "#ef4444", fontSize: 14,
              }}>
                ⚠️ {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || (mode === "upload" ? !previewFile : !url.trim())}
              style={{
                marginTop: 20,
                width: "100%", padding: "14px",
                background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                border: "none", borderRadius: 12,
                color: "#fff", fontWeight: 700, fontSize: 15,
                cursor: "pointer",
                opacity: (mode === "upload" ? !previewFile : !url.trim()) ? 0.5 : 1,
                transition: "opacity 0.2s",
              }}>
              🔬 Analyze Image
            </button>
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingScreen />}

        {/* Results Dashboard */}
        {report && (
          <div>
            {/* Top action bar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
              <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>
                Forensic Analysis Report
                <span style={{ fontSize: 13, color: "#4b5563", fontWeight: 400, marginLeft: 12 }}>
                  ID: {report.report_id}
                </span>
              </h2>
              <button onClick={reset} style={{
                padding: "8px 18px",
                background: "#1f2937", border: "1px solid #374151",
                borderRadius: 8, color: "#9ca3af",
                cursor: "pointer", fontSize: 13,
              }}>
                ← New Analysis
              </button>
            </div>

            {/* Misuse Warning */}
            {report.misuse_warning && (
              <div style={{
                padding: "12px 20px", marginBottom: 20,
                background: "#ef444415",
                border: "1px solid #ef444440",
                borderRadius: 12, color: "#fca5a5",
                fontSize: 14, fontWeight: 600,
              }}>
                {report.misuse_warning}
              </div>
            )}

            {/* Top Stats Row */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 16,
              marginBottom: 24,
            }}>
              {/* Prediction Badge */}
              <div style={{
                background: "#0d1117", border: `1px solid ${getPredColor(report.prediction)}40`,
                borderRadius: 16, padding: "20px 24px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
              }}>
                <div style={{ fontSize: 40 }}>
                  {report.prediction === "Fake" ? "🔴" : "🟢"}
                </div>
                <div style={{
                  fontSize: 24, fontWeight: 900,
                  color: getPredColor(report.prediction),
                }}>
                  {report.prediction}
                </div>
                <div style={{ fontSize: 13, color: "#6b7280" }}>Prediction</div>
              </div>

              {/* Confidence */}
              <div style={{
                background: "#0d1117", border: "1px solid #21262d",
                borderRadius: 16, padding: "20px 24px",
              }}>
                <div style={{ fontSize: 13, color: "#6b7280", marginBottom: 12 }}>
                  Confidence
                </div>
                <AnimatedBar
                  value={report.confidence}
                  color={getPredColor(report.prediction)}
                  label="Model confidence"
                />
                <div style={{
                  fontSize: 28, fontWeight: 900, marginTop: 4,
                  color: getPredColor(report.prediction),
                }}>
                  {report.confidence}%
                </div>
              </div>

              {/* Trust Score */}
              <div style={{
                background: "#0d1117", border: "1px solid #21262d",
                borderRadius: 16, padding: "20px 24px",
                display: "flex", flexDirection: "column", alignItems: "center",
              }}>
                <div style={{ width: 80, height: 80, marginBottom: 8 }}>
                  <CircularProgressbar
                    value={report.trust_score}
                    text={`${report.trust_score}`}
                    styles={buildStyles({
                      pathColor: getTrustColor(report.trust_score),
                      textColor: getTrustColor(report.trust_score),
                      trailColor: "#1f2937",
                      textSize: "22px",
                    })}
                  />
                </div>
                <div style={{ fontSize: 13, color: "#6b7280" }}>Trust Score</div>
              </div>

              {/* Risk Level */}
              <div style={{
                background: "#0d1117",
                border: `1px solid ${getRiskColor(report.risk_level)}40`,
                borderRadius: 16, padding: "20px 24px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
              }}>
                <div style={{ fontSize: 36 }}>
                  {report.risk_level === "High" ? "🚨" : report.risk_level === "Medium" ? "⚠️" : "✅"}
                </div>
                <div style={{
                  fontSize: 22, fontWeight: 900,
                  color: getRiskColor(report.risk_level),
                }}>
                  {report.risk_level}
                </div>
                <div style={{ fontSize: 13, color: "#6b7280" }}>Risk Level</div>
              </div>
            </div>

            {/* Visual Analysis Tabs */}
            <div style={{
              background: "#0d1117", border: "1px solid #21262d",
              borderRadius: 20, marginBottom: 24, overflow: "hidden",
            }}>
              <div style={{
                display: "flex", gap: 0,
                borderBottom: "1px solid #21262d",
              }}>
                {[
                  { id: "heatmap",    label: "🔥 Heatmap" },
                  { id: "regions",    label: "🎯 Regions" },
                  { id: "fft",        label: "📡 Frequency" },
                ].map(tab => (
                  <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                    style={{
                      padding: "14px 20px",
                      border: "none",
                      background: activeTab === tab.id ? "#161b22" : "transparent",
                      color: activeTab === tab.id ? "#a5b4fc" : "#6b7280",
                      fontWeight: activeTab === tab.id ? 700 : 400,
                      cursor: "pointer",
                      fontSize: 14,
                      borderBottom: activeTab === tab.id ? "2px solid #6366f1" : "2px solid transparent",
                      transition: "all 0.2s",
                    }}>
                    {tab.label}
                  </button>
                ))}
              </div>

              <div style={{ padding: 24 }}>
                {activeTab === "heatmap" && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        Grad-CAM++ Activation Overlay
                      </p>
                      <HeatmapSlider
                        original={report.annotated_image}
                        heatmap={report.heatmap_image}
                      />
                    </div>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        Score Components
                      </p>
                      {Object.entries({
                        "Model Confidence":   report.score_components?.model_confidence,
                        "Heatmap Intensity":  report.score_components?.heatmap_intensity,
                        "FFT Anomaly":        report.score_components?.fft_anomaly,
                        "Caption Severity":   report.score_components?.caption_severity,
                      }).map(([label, val]) => (
                        <AnimatedBar key={label} label={label} value={val ?? 0}
                          color="#6366f1" />
                      ))}
                      <div style={{ marginTop: 16 }}>
                        <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>
                          Suspicious Regions Detected
                        </div>
                        <div style={{ fontSize: 28, fontWeight: 800, color: "#a5b4fc" }}>
                          {report.suspicious_regions?.length ?? 0}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === "regions" && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        Annotated Regions
                      </p>
                      <img
                        src={`data:image/png;base64,${report.annotated_image}`}
                        alt="Annotated"
                        style={{ width: "100%", borderRadius: 12 }}
                      />
                    </div>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        Region Analysis
                      </p>
                      {report.suspicious_regions?.length ? (
                        report.suspicious_regions.map((reg, i) => (
                          <RegionCard key={i} region={reg} index={i} />
                        ))
                      ) : (
                        <div style={{ color: "#4b5563", fontSize: 14, padding: 20, textAlign: "center" }}>
                          No suspicious regions detected.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === "fft" && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        FFT Frequency Spectrum
                      </p>
                      <img
                        src={`data:image/png;base64,${report.fft_image}`}
                        alt="FFT Spectrum"
                        style={{ width: "100%", borderRadius: 12 }}
                      />
                    </div>
                    <div>
                      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
                        Frequency Analysis
                      </p>
                      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                        <div style={{
                          background: "#161b22",
                          border: "1px solid #30363d",
                          borderRadius: 12, padding: 16,
                        }}>
                          <div style={{ fontSize: 12, color: "#6b7280" }}>High-Frequency Energy Ratio</div>
                          <div style={{
                            fontSize: 28, fontWeight: 800, marginTop: 4,
                            color: report.fft_analysis?.high_freq_energy > 0.85 ? "#ef4444" : "#22c55e",
                          }}>
                            {((report.fft_analysis?.high_freq_energy ?? 0) * 100).toFixed(1)}%
                          </div>
                          <div style={{ fontSize: 11, color: "#4b5563", marginTop: 4 }}>
                            Natural images: ~70–85% | GAN images: often >85%
                          </div>
                        </div>
                        <div style={{
                          background: "#161b22",
                          border: "1px solid #30363d",
                          borderRadius: 12, padding: 16,
                        }}>
                          <div style={{ fontSize: 12, color: "#6b7280" }}>FFT Anomaly Score</div>
                          <AnimatedBar
                            value={report.fft_analysis?.fft_anomaly_score ?? 0}
                            color="#6366f1"
                            label=""
                          />
                          <div style={{
                            fontSize: 28, fontWeight: 800,
                            color: (report.fft_analysis?.fft_anomaly_score ?? 0) > 50
                              ? "#ef4444" : "#22c55e",
                          }}>
                            {report.fft_analysis?.fft_anomaly_score ?? 0} / 100
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Forensic Explanation */}
            <div style={{
              background: "#0d1117", border: "1px solid #21262d",
              borderRadius: 20, padding: 28, marginBottom: 24,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 8,
                  background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16,
                }}>🧠</div>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>
                  AI Forensic Explanation
                </h3>
              </div>
              <p style={{
                color: "#d1d5db", lineHeight: 1.8, fontSize: 15, margin: 0,
                borderLeft: "3px solid #6366f1",
                paddingLeft: 16,
              }}>
                {report.forensic_explanation}
              </p>
            </div>

            {/* Timestamp */}
            <div style={{ textAlign: "center", fontSize: 12, color: "#374151" }}>
              Analysis completed: {new Date(report.timestamp).toLocaleString()} ·
              Report ID: {report.report_id}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

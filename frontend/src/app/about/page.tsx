export default function AboutPage() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold mb-4">About DeepScan AI</h1>
      <p className="text-[var(--muted)] mb-6">
        DeepScan AI is an explainable deepfake detection platform designed for research demonstration,
        academic projects, and portfolio showcase. It combines deep learning classification with
        interpretable AI techniques and digital forensic analysis.
      </p>

      <div className="grid gap-6 md:grid-cols-2 not-prose">
        <div className="card">
          <h2 className="font-semibold mb-2">Detection Model</h2>
          <p className="text-sm text-[var(--muted)]">
            MobileNetV2 backbone fine-tuned for binary real/fake classification on 224×224 RGB images.
          </p>
        </div>
        <div className="card">
          <h2 className="font-semibold mb-2">Explainability</h2>
          <p className="text-sm text-[var(--muted)]">
            Grad-CAM++ generates activation heatmaps. Suspicious regions are extracted and captioned
            using BLIP (optional) or rule-based analysis.
          </p>
        </div>
        <div className="card">
          <h2 className="font-semibold mb-2">Forensic Engine</h2>
          <p className="text-sm text-[var(--muted)]">
            FFT spectrum analysis, noise uniformity, texture sharpness, and JPEG compression artifact detection.
          </p>
        </div>
        <div className="card">
          <h2 className="font-semibold mb-2">Social Safety</h2>
          <p className="text-sm text-[var(--muted)]">
            Risk assessment for cyberbullying, misinformation, impersonation, and identity abuse scenarios.
          </p>
        </div>
      </div>
    </div>
  );
}

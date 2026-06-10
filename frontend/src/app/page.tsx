import Link from "next/link";
import { Shield, Scan, Brain, FileSearch, Zap } from "lucide-react";

const features = [
  { icon: Scan, title: "Deepfake Detection", desc: "MobileNetV2 classifier fine-tuned for real vs fake classification." },
  { icon: Brain, title: "Explainable AI", desc: "Grad-CAM++ heatmaps highlight suspicious facial regions." },
  { icon: FileSearch, title: "Forensic Analysis", desc: "FFT, noise, texture, and compression artifact analysis." },
  { icon: Zap, title: "AI Explanations", desc: "Evidence-based forensic narratives powered by Claude or rule-based fallback." },
];

export default function HomePage() {
  return (
    <div className="animate-fade-in">
      <section className="py-16 text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-500/30 bg-brand-500/10 px-4 py-1.5 text-sm text-brand-500">
          <Shield className="h-4 w-4" /> Production-Ready XAI Platform
        </div>
        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-6xl">
          Explainable <span className="text-brand-500">Deepfake</span> Detection
        </h1>
        <p className="mx-auto mb-8 max-w-2xl text-lg text-[var(--muted)]">
          Upload an image or analyze a URL to detect manipulated media with Grad-CAM++ explainability,
          forensic frequency analysis, and social media safety risk assessment.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link href="/upload" className="btn-primary">Upload Image</Link>
          <Link href="/analyze" className="rounded-lg border border-[var(--border)] px-5 py-2.5 text-sm font-semibold hover:bg-brand-500/10">
            Analyze URL
          </Link>
        </div>
      </section>

      <section className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {features.map((f) => (
          <div key={f.title} className="card animate-slide-up">
            <f.icon className="mb-3 h-8 w-8 text-brand-500" />
            <h3 className="mb-2 font-semibold">{f.title}</h3>
            <p className="text-sm text-[var(--muted)]">{f.desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
}

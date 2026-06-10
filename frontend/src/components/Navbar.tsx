"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Moon, Sun, Shield } from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { clsx } from "clsx";

const links = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Upload" },
  { href: "/analyze", label: "Analyze URL" },
  { href: "/reports", label: "Reports" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--card)]/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
        <Link href="/" className="flex items-center gap-2 font-bold text-lg">
          <Shield className="h-6 w-6 text-brand-500" />
          <span>DeepScan AI</span>
        </Link>
        <nav className="hidden md:flex items-center gap-6">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={clsx(
                "text-sm font-medium transition hover:text-brand-500",
                pathname === l.href ? "text-brand-500" : "text-[var(--muted)]"
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <button
          onClick={toggle}
          aria-label="Toggle theme"
          className="rounded-lg border border-[var(--border)] p-2 hover:bg-brand-500/10"
        >
          {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </button>
      </div>
    </header>
  );
}

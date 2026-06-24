// Implements SPEC-0027 (Tailwind config, ADR-0004 stack).
// Console design tokens (SPEC-0027 SS 4: dense dark operator surface).
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Graphite surface hierarchy (darkest -> lightest).
        ink: {
          950: "#070a0e",
          900: "#0b0f15",
          850: "#0f141c",
          800: "#141a24",
          700: "#1b232f",
          600: "#26303f",
        },
        line: {
          DEFAULT: "#222c3a",
          soft: "#19212d",
          strong: "#303c4e",
        },
        fg: {
          DEFAULT: "#e7edf4",
          muted: "#93a1b5",
          faint: "#5d6b7e",
        },
        // Single teal accent for action + ready state.
        accent: {
          DEFAULT: "#2dd4bf",
          strong: "#5eead4",
          soft: "#0e3b37",
        },
        ok: "#34d399",
        warn: "#fbbf24",
        bad: "#f87171",
      },
      fontFamily: {
        sans: [
          '"IBM Plex Sans"',
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          '"IBM Plex Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 28px -16px rgba(0,0,0,0.8)",
        modal: "0 24px 80px -24px rgba(0,0,0,0.85)",
        glow: "0 0 0 1px rgba(45,212,191,0.35), 0 0 24px -6px rgba(45,212,191,0.45)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "translateY(6px) scale(0.985)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
      },
      animation: {
        shimmer: "shimmer 1.4s linear infinite",
        "fade-in": "fade-in 0.18s ease-out",
        "scale-in": "scale-in 0.18s cubic-bezier(0.16,1,0.3,1)",
        "pulse-soft": "pulse-soft 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

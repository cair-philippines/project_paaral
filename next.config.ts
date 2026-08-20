import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pure static export — no Next.js server at runtime (decided 2026-08-20,
  // see .claude/rules/memory-decisions.md). Deploys to classic Firebase
  // Hosting as a plain static-file CDN.
  output: "export",
  images: {
    // next/image's optimization API needs a server; static export has none.
    unoptimized: true,
  },
};

export default nextConfig;

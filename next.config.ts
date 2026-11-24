import type { NextConfig } from "next";

/**
 * Netlify deployment: netlify.toml sets NEXT_STATIC_EXPORT=1 so `npm run build` produces a fully static
 * site in the `out/` folder (no server needed). Locally / in the preview sandbox the normal build is used.
 */
const staticExport = process.env.NEXT_STATIC_EXPORT === "1";

const nextConfig: NextConfig = {
  ...(staticExport ? { output: "export" as const } : {}),
  images: { unoptimized: true },
  trailingSlash: false,
};

export default nextConfig;

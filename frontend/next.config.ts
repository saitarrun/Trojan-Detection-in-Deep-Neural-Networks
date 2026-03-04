import type { NextConfig } from "next";

// Based on your URL: https://csuf-titans.nrp-nautilus.io/user/saitarrunpitta@csu.fullerton.edu/lab
// The proxy prefix is everything before /lab
const prefix = "/user/saitarrunpitta@csu.fullerton.edu/proxy/3000";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,

  // These are critical for JupyterHub Proxy to load CSS/JS correctly
  basePath: prefix,
  assetPrefix: prefix,
  trailingSlash: true,

  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/health',
        destination: 'http://localhost:8000/health',
      }
    ];
  },
};

export default nextConfig;

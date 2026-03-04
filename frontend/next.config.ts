import type { NextConfig } from "next";

// The proxy prefix from your URL: https://csuf-titans.nrp-nautilus.io/user/saitarrunpitta@csu.fullerton.edu/lab
const prefix = "/user/saitarrunpitta@csu.fullerton.edu/proxy/3000";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,

  // We REMOVE basePath because it causes 404s on the proxy.
  // We KEEP assetPrefix to ensure CSS/JS load from the correct subpath.
  assetPrefix: prefix,

  // This helps the proxy handle sub-routes
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

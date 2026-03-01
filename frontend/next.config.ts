import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://api:8000/api/:path*',
      },
      {
        source: '/health',
        destination: 'http://api:8000/health',
      },
      // Keep support for raw localhost for dev mode outside docker
      {
        source: '/local-api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      }
    ];
  },
};

export default nextConfig;

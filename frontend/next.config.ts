import type { NextConfig } from "next";

const isNautilus = typeof process !== 'undefined' && process.env.JUPYTERHUB_SERVICE_PREFIX;
const basePath = isNautilus ? `${process.env.JUPYTERHUB_SERVICE_PREFIX}proxy/3000` : '';

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  basePath: basePath,
  assetPrefix: basePath,
  async rewrites() {
    const apiDest = isNautilus
      ? `${process.env.JUPYTERHUB_SERVICE_PREFIX}proxy/8000`
      : 'http://localhost:8000';

    return [
      {
        source: '/api/:path*',
        destination: `${apiDest}/api/:path*`,
      },
      {
        source: '/health',
        destination: `${apiDest}/health`,
      }
    ];
  },
};

export default nextConfig;

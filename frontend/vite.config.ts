import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Single-port architecture: in dev, Vite serves the SPA and proxies /api to
// the FastAPI backend so the browser only ever talks to one origin
// (http://127.0.0.1:5173 → /api → 127.0.0.1:8000). In prod the SPA is built
// to dist/ and FastAPI mounts it directly.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});

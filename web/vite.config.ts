import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The client calls /api/... with no host, and the dev server forwards it to
// FastAPI. Same-origin in development as it would be in a deployed build, so
// there is no CORS special case to remember and nothing to change when shipping.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});

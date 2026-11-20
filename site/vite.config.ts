import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
  },
  server: {
    // 5173 (Vite's default) is used by another local project.
    port: 8080,
    // Fail with a clear error instead of silently picking another port.
    strictPort: true,
  },
});

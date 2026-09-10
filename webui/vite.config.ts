/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../web/static',
    emptyOutDir: true,
  },
  // Tests share this file's plugins and resolution deliberately: a component
  // that builds must also be the component under test, and a second transform
  // config is the usual way that stops being true.
  test: {
    // Components here read `localStorage`, `matchMedia` and `EventSource`, so
    // they need a DOM rather than bare Node.
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    // Only our own tests -- without this the default glob also walks
    // node_modules and tries to run other packages' suites.
    include: ['src/**/*.test.{ts,tsx}'],
  },
})

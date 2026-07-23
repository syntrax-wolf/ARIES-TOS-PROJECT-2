import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy the leaderboard API so the app can just call /api/... in dev —
    // same origin, so no CORS dance and no base-URL env var to keep in sync.
    proxy: {
      '/api': {
        target: process.env.SYLVA_API ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

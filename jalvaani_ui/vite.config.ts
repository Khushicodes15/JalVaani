import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    // Proxy API calls to FastAPI during development
    proxy: {
      '/predict':  { target: 'http://localhost:8000', changeOrigin: true },
      '/forecast': { target: 'http://localhost:8000', changeOrigin: true },
      '/report':   { target: 'http://localhost:8000', changeOrigin: true },
      '/stations': { target: 'http://localhost:8000', changeOrigin: true },
      '/stats':    { target: 'http://localhost:8000', changeOrigin: true },
      '/health':   { target: 'http://localhost:8000', changeOrigin: true },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    // Code splitting: users only download what each page needs
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react':   ['react', 'react-dom', 'react-router-dom'],
          'vendor-query':   ['@tanstack/react-query'],
          'vendor-charts':  ['recharts'],
          'vendor-icons':   ['lucide-react'],
        },
      },
    },
  },
})

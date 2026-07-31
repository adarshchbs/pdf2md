import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({ plugins: [svelte(), tailwindcss()], build: { outDir: 'dist' }, server: { port: 5173, proxy: { '/comparison': 'http://localhost:8010' } } });

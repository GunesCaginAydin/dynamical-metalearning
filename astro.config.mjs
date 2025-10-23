// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
// https://astro.build/config
// export default defineConfig({});

export default defineConfig({
  site: 'https://GunesCaginAydin.github.io',
  base: '/',
  vite: {
    plugins: [tailwindcss()],
  },
})

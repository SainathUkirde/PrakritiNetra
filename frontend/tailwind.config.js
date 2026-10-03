/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0f1117',
        surface: '#1a1f2e',
        border: 'rgba(255,255,255,0.10)',
      },
    },
  },
  plugins: [],
}

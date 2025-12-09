/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#181a1f',
        'dark-panel': '#20232a',
        'dark-header': '#1a1d23',
      },
    },
  },
  plugins: [],
}

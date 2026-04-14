/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        traitor: "#ef4444",
        scientist: "#3b82f6",
        auditor: "#a855f7",
        jury: "#f59e0b",
      },
    },
  },
  plugins: [],
};

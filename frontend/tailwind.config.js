/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"] ,
  theme: {
    extend: {
      colors: {
        ink: "#0b0d12",
        fog: "#e9ecef",
        haze: "#c4c7cf",
        ember: "#f97316",
        sea: "#0ea5e9",
        moss: "#22c55e",
      },
      boxShadow: {
        glow: "0 0 40px rgba(14, 165, 233, 0.25)",
        ember: "0 0 35px rgba(249, 115, 22, 0.35)",
      },
      animation: {
        float: "float 8s ease-in-out infinite",
        pulseSlow: "pulseSlow 6s ease-in-out infinite",
        rise: "rise 700ms ease-out both",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-16px)" },
        },
        pulseSlow: {
          "0%, 100%": { opacity: 0.25 },
          "50%": { opacity: 0.6 },
        },
        rise: {
          "0%": { opacity: 0, transform: "translateY(16px)" },
          "100%": { opacity: 1, transform: "translateY(0px)" },
        },
      },
    },
  },
  plugins: [],
};

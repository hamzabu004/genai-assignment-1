import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      borderRadius: {
        none: "0px",
        DEFAULT: "0px",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "SF Pro Display",
          "SF Pro Text",
          "system-ui",
          "sans-serif",
        ],
      },
      colors: {
        accent: "#0071E3",
        surface: "#F5F5F7",
        border: "#D2D2D7",
        ink: "#1D1D1F",
        muted: "#6E6E73",
        success: "#1F8A3B",
        danger: "#D93025",
        warning: "#B8860B",
      },
    },
  },
  plugins: [],
};
export default config;

import { createTheme } from "@mui/material/styles";

// Palette matches the CSS custom properties in globals.css (2026-08-24
// recolor — see SKILLS.md "Design System" for the pending write-up once
// Paula confirms the Step 2 structural template). Kept in sync manually
// since MUI's theme can't read CSS variables for contrast calculations.
const theme = createTheme({
  palette: {
    primary: {
      main: "#19266b",
    },
    secondary: {
      main: "#fcca81",
    },
    error: {
      main: "#b23836",
    },
    text: {
      primary: "#020315",
    },
    background: {
      default: "#fcfcfd",
    },
  },
  typography: {
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif",
  },
});

export default theme;

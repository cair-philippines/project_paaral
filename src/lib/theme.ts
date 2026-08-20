import { createTheme } from "@mui/material/styles";

// Starting palette only — matches the existing design tokens documented in
// SKILLS.md ("Design System"). Tailwind Plus templates drive the actual
// flat-color layout/responsiveness; this theme just gives MUI components a
// sensible baseline until that visual pass happens.
const theme = createTheme({
  palette: {
    primary: {
      main: "#1a4b8c", // DepEd blue
    },
  },
  typography: {
    fontFamily: "var(--font-geist-sans), Arial, Helvetica, sans-serif",
  },
});

export default theme;

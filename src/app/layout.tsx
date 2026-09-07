import type { Metadata } from "next";
import { Geist, Geist_Mono, Public_Sans, Lato } from "next/font/google";
import ThemeRegistry from "@/components/templates/ThemeRegistry";
import { ApplicationStateProvider } from "@/components/templates/ApplicationStateProvider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Scoped to the landing hero only (2026-09-07) — not a sitewide change, and a
// deliberate, explicit departure from DESIGN.md's "One Typeface Rule" for
// that one surface. Geist stays the default everywhere else.
const publicSans = Public_Sans({
  variable: "--font-public-sans",
  subsets: ["latin"],
  weight: ["300", "800"],
});

const lato = Lato({
  variable: "--font-lato",
  subsets: ["latin"],
  weight: ["300"],
});

export const metadata: Metadata = {
  title: "PAARAL — Student View",
  description:
    "Educational Service Contracting (ESC) enrollment portal for Grade 6 to Grade 7 learners.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${publicSans.variable} ${lato.variable} h-full scroll-smooth antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <ThemeRegistry>
          <ApplicationStateProvider>{children}</ApplicationStateProvider>
        </ThemeRegistry>
      </body>
    </html>
  );
}

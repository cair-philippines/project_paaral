import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
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

export const metadata: Metadata = {
  title: "PAARAL — Student View",
  description:
    "Educational Service Contracting (ESC) enrollment portal for Grade 6 to Grade 7 learners.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <ThemeRegistry>
          <ApplicationStateProvider>{children}</ApplicationStateProvider>
        </ThemeRegistry>
      </body>
    </html>
  );
}

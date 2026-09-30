import type { Metadata } from "next";
import { Sidebar } from "@/components/layout/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Precision Vision Studio — Neural Image Restoration Suite",
  description:
    "Scientific instrument environment for neural image restoration, classification gating, mixture-of-experts, and face-to-sketch synthesis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full bg-[#FFFFFF]">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block"
        />
      </head>
      <body className="min-h-full flex text-[#1D1D1F] bg-[#FFFFFF] antialiased">
        <Sidebar />
        <div className="pl-[240px] w-full min-h-screen bg-[#FFFFFF]">
          <main className="max-w-[1240px] mx-auto p-6 lg:p-8 min-h-screen">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}

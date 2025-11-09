import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CCTV MJPEG Stream",
  description: "Real-time MJPEG stream viewer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

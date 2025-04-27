// src/app/layout.tsx

import { Geist, Geist_Mono } from "next/font/google";
import "../styles/globals.css";
import 'bootstrap/dist/css/bootstrap.min.css';
import { AppWrapper } from "./AppWrapper";
import { AuthProvider } from "@/context/AuthContext";
import Image from "next/image";

// Load fonts in module scope
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const fontsClassName = `${geistSans.variable} ${geistMono.variable} antialiased`

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div style={{
          position: "fixed",
          top: 0,
          bottom: 0,
          left: 0,
          width: "60px",
          zIndex: 0,
          pointerEvents: "none",
        }}>
          <Image
            src="/left.png"
            alt="left decoration"
            fill
            style={{ objectFit: 'cover', objectPosition: 'top left' }}
            priority
          />
        </div>
        <div style={{
          position: "fixed",
          top: 0,
          bottom: 0,
          right: 0,
          width: "60px",
          zIndex: 0,
          pointerEvents: "none"
        }}>
          <Image
            src="/right.png"
            alt="right decoration"
            fill
            style={{ objectFit: 'cover', objectPosition: 'top right' }}
          />
        </div>
        <AuthProvider>
          <AppWrapper>
            {children}
          </AppWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}

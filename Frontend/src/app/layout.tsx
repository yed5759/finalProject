// src/app/layout.tsx


import "../styles/globals.css";
import 'bootstrap/dist/css/bootstrap.min.css';
import { AppWrapper } from "@/components/AppWrapper";
import { AuthProvider } from "@/context/AuthContext";
import React from "react";

export const metadata = {
  title: 'Taking Notes',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="left-decoration" />
        <div className="right-decoration" />
        <AuthProvider>
          <AppWrapper>
            {children}
          </AppWrapper>
        </AuthProvider>
      </body>
    </html>
  );
}

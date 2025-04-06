'use client';
import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";

export default function AuthWrapper({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    console.log("Token found:", token);
    setIsAuthenticated(!!token);
  }, []);

  useEffect(() => {
    // If no user is authenticated and we're not on the /landing route, redirect to /landing
    if (isAuthenticated === false && pathname !== '/landing') {
      router.push("/landing");
    }
  }, [isAuthenticated, router, pathname]);

  if (isAuthenticated === null) {
    return <div>Loading...</div>;
  }

  // If user is not authenticated but on the landing page, render the children to show the landing page
  if (!isAuthenticated && pathname === '/landing') {
    return <>{children}</>;
  }

  // If not authenticated and not on the landing route, render nothing — redirection has already been made
  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
}

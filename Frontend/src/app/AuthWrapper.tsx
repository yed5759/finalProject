'use client';
import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";

export default function AuthWrapper({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("accessToken");
      console.log("[AuthWrapper] Token found:", token);
      setIsAuthenticated(!!token);  // Update the authentication status based on the token
    }
  }, []);  // This effect runs only once when the component mounts

  // After isAuthenticated changes, we perform the necessary redirection
  useEffect(() => {
    console.log("[AuthWrapper] isAuthenticated:", isAuthenticated);
    console.log("[AuthWrapper] pathname:", pathname);

    // If the user is not authenticated and we're not on the landing page, redirect to landing
    //if (isAuthenticated === false && pathname !== '/') {
      //console.log("[AuthWrapper] Redirecting to /landing");
      //router.push("/");
    //}
    // If the user is authenticated and we're on the landing page, redirect to /home
    if (isAuthenticated === true && pathname === '/') {
      console.log("[AuthWrapper] Redirecting to /home");
      router.push("/home");
    }
  }, [isAuthenticated, pathname, router]);  // This effect runs when isAuthenticated or pathname changes

  if (isAuthenticated === null) {
    return <div>Loading...</div>;  // Show a loading state until the authentication status is determined
  }

  return <>{children}</>;
}

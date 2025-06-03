// src/app/AppWrapper.tsx

'use client';

import type { PropsWithChildren } from "react";
import { useEffect } from "react";
import Navbar from "@/components/Navbar";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";

export const AppWrapper = ({ children }: PropsWithChildren) => {
    const { isAuthenticated, checkAuth } = useAuth();
    const pathname = usePathname();
    const router = useRouter();

    useEffect(() => {
        checkAuth();

        // Redirect based on auth status
        if (isAuthenticated && pathname === "/Landing") {
            router.replace("/Home");
        } else if (!isAuthenticated && pathname.startsWith("/Home")) {
            router.replace("/Landing");
        }
    }, [pathname, isAuthenticated, checkAuth, router]);

    // Don't render anything until we know if the user is authenticated
    if (isAuthenticated === null) return null;

    const showNavbarPages = ['/Home', '/MyLibrary', '/Notes'];
    const showNavbar = isAuthenticated && showNavbarPages.includes(pathname);

    return (
        <div className="flex" style={{ width: "100%" }}>
            {showNavbar && <Navbar />}
            <main className="flex-1">
                {children}
            </main>
        </div>
    );
};

// src/app/AppWrapper.tsx

'use client';

import {PropsWithChildren} from "react";
import { useEffect } from "react";
import Navbar from "@/components/Navbar";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";

export const AppWrapper = ({ children }: PropsWithChildren) => {
    const { isAuthenticated, checkAuth, loading} = useAuth();
    const pathname = usePathname();
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        if(loading) return;
        // Redirect based on auth status
        if (isAuthenticated && pathname === "/landing") {
            router.replace("/home");
        } else if (!isAuthenticated && !pathname.startsWith("/landing")) {
            router.replace("/landing");
        }
    }, [pathname, isAuthenticated, loading, router]);

    // Don't render anything until we know if the user is authenticated
    if (isAuthenticated === null) return null;

    const showNavbarPages = ['/home', '/MyLibrary', '/Notes'];
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

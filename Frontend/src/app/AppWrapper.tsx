// src/app/AppWrapper.tsx

'use client';

import type { PropsWithChildren } from "react";
import { useEffect, useState } from "react";
import Navbar from "@/components/Navbar";
import { isAuthenticated } from "@/utils/cognito";

export const AppWrapper = ({ children }: PropsWithChildren) => {
    const [showNavbar, setShowNavbar] = useState<boolean>(false);

    useEffect(() => {
        setShowNavbar(isAuthenticated());
    }, []);

    return (
        <div className="flex" style={{ width: "100%" }}>
            {showNavbar && <Navbar />}
            <main className="flex-1">
                {children}
            </main>
        </div>
    )
}
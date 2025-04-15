'use client';
import type { PropsWithChildren } from "react";

import Navbar from "@/components/Navbar";
import { usePathname } from "next/navigation";

export default function AppWrapper({ children }: PropsWithChildren) {
    const pathname = usePathname();
    const showNavbar = pathname !== '/';

    return (
        <div className="flex" style={{ width: "100%" }}>
            {showNavbar && <Navbar />}
            <main className="flex-1">
                {children}
            </main>
        </div>
    );
}
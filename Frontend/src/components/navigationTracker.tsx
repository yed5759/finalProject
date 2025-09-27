'use client';

import { useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";

export function NavigationTracker() {
    const pathname = usePathname();
    const searchParams = useSearchParams();

    useEffect(() => {
        if (typeof window !== "undefined" && pathname) {
            const fullPath =
                pathname + (searchParams?.toString() ? `?${searchParams}` : "");
            const stack = JSON.parse(sessionStorage.getItem("navStack") || "[]");

            if (stack[stack.length - 1] !== fullPath) {
                stack.push(fullPath);
                sessionStorage.setItem("navStack", JSON.stringify(stack));
            }
        }
    }, [pathname, searchParams]);

    return null;
}
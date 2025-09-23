//src/app/page.tsx

'use client'

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';


export function NavigationTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (typeof window !== "undefined" && pathname) {
      const fullPath =
        pathname + (searchParams?.toString() ? `?${searchParams}` : "");
      const stack = JSON.parse(sessionStorage.getItem("navStack") || "[]");

      // push only if it's not the same as the last entry
      if (stack[stack.length - 1] !== fullPath) {
        stack.push(fullPath);
        sessionStorage.setItem("navStack", JSON.stringify(stack));
      }
    }
  }, [pathname, searchParams]);

  return null;
}

export default function Page() {
  const { isAuthenticated, loading, checkAuth } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    const initAuth = async () => {
      const code = searchParams.get('code');
      await checkAuth(code || undefined);
      setAuthChecked(true);
    };
    initAuth();
  }, []);

  useEffect(() => {
    if (!loading && authChecked) {
      if (isAuthenticated) {
        router.replace('/home');
      } else {
        router.replace('/landing');
      }
    }
  }, [isAuthenticated, loading, authChecked, router]);

  if (loading) {
    return <></>;
  }

  return null;
}


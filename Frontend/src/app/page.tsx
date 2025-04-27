//src/app/page.tsx

'use client'

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';

const Page = () => {
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
    return <>Loading...</>;
  }

  return null;
};

export default Page;

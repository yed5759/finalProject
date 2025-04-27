//src/app/page.tsx

'use client'

import { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';

const Page = () => {
  const { isAuthenticated, loading, checkAuth } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const initAuth = async () => {
      const code = searchParams.get('code');
      await checkAuth(code || undefined);
    };
    initAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!loading) {
      if (isAuthenticated) {
        router.replace('/home');
      } else {
        router.replace('/landing');
      }
    }
  }, [isAuthenticated, loading, router]);

  if (loading) {
    return <>Loading...</>;
  }

  return null;
};

export default Page;

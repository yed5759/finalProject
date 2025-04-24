'use client'

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { isAuthenticated, exchangeCodeForToken } from '@/utils/cognito';

const Page = () => {
  const [authState, setAuthState] = useState<boolean | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const checkAuth = async () => {
      const tokenExists = isAuthenticated();
      const code = searchParams.get('code');

      if (tokenExists) {
        setAuthState(true);
        router.push('/home');
      } else if (code) {
        const success = await exchangeCodeForToken(code);
        setAuthState(success);
        router.push(success ? '/home' : '/landing');
      } else {
        setAuthState(false);
        router.push('/landing');
      }
    };

    checkAuth();
  }, [searchParams, router]);

  if (isAuthenticated === null) {
    return <div>Loading...</div>; // Show loading screen while checking the authentication
  }

  return null; // No content to render here
};

export default Page;

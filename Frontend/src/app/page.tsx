'use client'

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { parseCookies } from 'nookies';

const Page = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const cookies = parseCookies();
    const accessToken = cookies.accessToken;
    const code = searchParams.get('code');

    const authenticate = async () => {
      if (accessToken) {
        setIsAuthenticated(true);
        router.push('/home');
      } else if (code) {
        try {
          // Send code to the Flask server to exchange it for tokens
          const res = await fetch(`http://localhost:5000/auth/callback?code=${code}`, {
            method: 'GET',
            credentials: 'include',
          });

          if (!res.ok) throw new Error('Failed to exchange code for tokens');

          // After tokens are set in cookies, mark as authenticated
          setIsAuthenticated(true);
          router.push('/home');
        } catch (err) {
          // In case of an error, consider the user not authenticated
          setIsAuthenticated(false);
          router.push('/landing');
        }

      } else {
        setIsAuthenticated(false); // No code and no token, so not authenticated
        router.push('/landing');
      }
    };

    authenticate();
  }, [searchParams, router])

  if (isAuthenticated === null) {
    return <div>Loading...</div>; // Show loading screen while checking the authentication
  }

  return null; // No content to render here
};

export default Page;

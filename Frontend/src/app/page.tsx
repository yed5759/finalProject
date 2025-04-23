'use client'

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation'; // Use Next.js Router for navigation
import { parseCookies } from 'nookies';

const Page = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const cookies = parseCookies();
    const accessToken = cookies.accessToken;
    const code = searchParams.get('code');

    console.log('🟢 [page.tsx] Token from cookies:', accessToken);
    console.log('🔵 [page.tsx] Code from URL =', code);


    const authenticate = async () => {
      if (accessToken) {
        setIsAuthenticated(true);
        router.push('/home');
      } else if (code) {
        try {
          // Send code to the Flask server to exchange it for tokens
          console.log(`[page.tsx] sending fetch with code=${code}`)
          console.log(code)
          const res = await fetch(`http://localhost:5000/auth/callback?code=${code}`, {
            method: 'GET',
            credentials: 'include',
          });
          console.log('[page.tsx] has res from fetch with code')

          if (!res.ok) throw new Error('Failed to exchange code for tokens');
          // After tokens are set in cookies, mark as authenticated

          console.log('✅ [page.tsx] Token request successful');
          setIsAuthenticated(true);
          router.push('/home');
        } catch (err) {
          console.error('❌ [page.tsx] Error sending code to server:', err);
          setIsAuthenticated(false);
          // In case of an error, consider the user not authenticated
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


// useEffect(() => {
//   // Here, we're checking if the user is authenticated, using a token or any other authentication mechanism
//   const cookies = parseCookies();
//   const accessToken = cookies.accessToken; // קבל את הטוקן מה-cookies
//   console.log('🟢 [page.tsx] Token from cookies =', accessToken);


//   // const token = localStorage.getItem('accessToken'); // Replace with your token check
//   // console.log('🟢 [page.tsx] token =', token);
//   // if (token) {
//   if (accessToken) {
//     setIsAuthenticated(true);
//   } else {
//     setIsAuthenticated(false);
//   }
// }, []);

//   useEffect(() => {
//     console.log('🟠 [page.tsx] isAuthenticated =', isAuthenticated);
//     if (isAuthenticated === true) {
//       console.log('➡️ [page.tsx] redirecting to /home');
//       router.push('/home'); // Navigate to the home page if the user is authenticated
//     } else if (isAuthenticated === false) {
//       console.log('➡️ [page.tsx] redirecting to /landing');
//       router.push('/landing'); // Navigate to the landing page if the user is not authenticated
//     }
//   }, [isAuthenticated, router]);

//   if (isAuthenticated === null) {
//     return <div>Loading...</div>; // Show loading screen while checking the authentication
//   }

//   return null; // No content to render here
// };

export default Page;

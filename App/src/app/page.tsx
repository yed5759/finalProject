'use client'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation'; // Use Next.js Router for navigation

const Page = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const router = useRouter();

  useEffect(() => {
    // Here, we're checking if the user is authenticated, using a token or any other authentication mechanism
    const token = localStorage.getItem('accessToken'); // Replace with your token check
    if (token) {
      setIsAuthenticated(true);
    } else {
      setIsAuthenticated(false);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated === true) {
      router.push('/home'); // Navigate to the home page if the user is authenticated
    } else if (isAuthenticated === false) {
      router.push('/landing'); // Navigate to the landing page if the user is not authenticated
    }
  }, [isAuthenticated, router]);

  if (isAuthenticated === null) {
    return <div>Loading...</div>; // Show loading screen while checking the authentication
  }

  return null; // No content to render here
};

export default Page;

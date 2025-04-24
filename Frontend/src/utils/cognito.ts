// src/utils/cognito.ts

import { parseCookies } from 'nookies';

export const getAccessTokenFromCookies = (): string | null => {
  const cookies = parseCookies();
  return cookies.accessToken || null;
};

export const isAuthenticated = (): boolean => {
  const token = getAccessTokenFromCookies();
  return !!token;
};

export const exchangeCodeForToken = async (code: string): Promise<boolean> => {
  try {
    // Send code to the Flask server to exchange it for tokens
    const res = await fetch(`http://localhost:5000/auth/callback?code=${code}`, {
      method: 'GET',
      credentials: 'include',
    });

    if (!res.ok) throw new Error('Failed to exchange code for token');

    return true;
  } catch (err) {
    console.error('Token exchange error:', err);
    return false;
  }
};

// src/utils/cognito.ts

// Utility to dispatch custom event
const dispatchTokenChange = () => {
  const event = new Event("access_token_changed");
  window.dispatchEvent(event);
};

// Function to get the access token from localStorage
export const getAccessTokenFromLocalStorage = (): string | null => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
};

// Function to check if the user is authenticated
export const isAuthenticated = (): boolean => {
  if (typeof window === "undefined") return false;
  const token = getAccessTokenFromLocalStorage();
  return !!token; // Returns true if the token exists, false otherwise
};

// Function to exchange the authorization code for tokens
export const exchangeCodeForToken = async (code: string): Promise<boolean> => {
  try {
    // Check if tokens already exists
    if (isAuthenticated()) {
      return true;
    }

    // If not, send code to the Flask server to exchange it for tokens
    const res = await fetch(`http://localhost:5000/auth/callback?code=${code}`, {
      method: 'GET',
      credentials: 'include',
    });

    if (!res.ok) throw new Error('Failed to exchange code for token');

    // Parse the response as JSON and store tokens in localStorage
    const data = await res.json();

    // Save the tokens in localStorage
    localStorage.setItem("id_token", data.id_token);
    localStorage.setItem("access_token", data.access_token);

    // Dispatch event to notify token change (after login)
    dispatchTokenChange();

    return true;
  } catch (err) {
    console.error('Token exchange error:', err);
    return false;
  }
};

// Function to log the user out by clearing localStorage
export const logout = (): void => {
  localStorage.removeItem("access_token");
  localStorage.removeItem("id_token");

  // Dispatch event to notify token change (after logout)
  dispatchTokenChange();

  // Redirect to landing page
  window.location.href = "/Landing";
};

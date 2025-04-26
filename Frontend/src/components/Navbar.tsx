'use client';


import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import '../styles/Navbar.css';
// import { signOut } from '@aws-amplify/auth'
import { logout } from '@/utils/cognito';
// import { isAuthenticated } from '@/utils/cognito';
import { useAuth } from '@/context/AuthContext';
// import { useAuthenticator } from "@aws-amplify/ui-react";

export default function Navbar() {
  const { isAuthenticated, loading } = useAuth();
  // const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const router = useRouter();

  if (loading || !isAuthenticated) return null;

  // useEffect(() => {
  //   // Check if the user is authenticated based on localStorage
  //   const checkAuthentication = () => {
  //   };
  //   setIsLoggedIn(isAuthenticated());

  //   checkAuthentication();
  // }, []);

  // const { signOut } = useAuthenticator();

  const handleLogout = async () => {
    logout();
  };

  // if (!isLoggedIn) return null;

  return (
    <header className="w-full bg-blue-900 text-white flex justify-between items-center">
      <nav className="navbar">
        <div className="navbar-title">🎵 Taking Notes</div>
        <button onClick={handleLogout} className="navbar-button">Logout</button>
      </nav>
    </header>
  );
}
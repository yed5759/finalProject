'use client';

import { useRouter } from 'next/navigation';
import '../styles/Navbar.css';
import { signOut } from '@aws-amplify/auth'

// import { useAuthenticator } from "@aws-amplify/ui-react";

export default function Navbar() {
  const router = useRouter();

  // const { signOut } = useAuthenticator();

  const handleLogout = async () => {
    try {
      await signOut();
      localStorage.removeItem('accessToken'); // אם אני שומר טוקן ידנית
      router.push('/landing'); // או כל עמוד אחר שתרצה להפנות אליו
    } catch (error) {
      console.error('Error signing out: ', error)
    }
  };

  return (
    <header className="w-full bg-blue-900 text-white flex justify-between items-center">
      <nav className="navbar">  {/* השתמש במחלקת ה־CSS 'navbar' */}
        <div className="navbar-title">🎵 Taking Notes</div> {/* השתמש במחלקת ה־CSS 'navbar-title' */}
        {/* <button onClick={signOut} className="navbar-button">Logout</button> */}
        <button onClick={handleLogout} className="navbar-button">Logout</button> {/* השתמש במחלקת ה־CSS 'navbar-button' */}
      </nav>
    </header>
  );
}

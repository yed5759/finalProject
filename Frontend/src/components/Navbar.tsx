'use client';


import '../styles/Navbar.css';
import { logout } from '@/utils/cognito';
import { useAuth } from '@/context/AuthContext';

export default function Navbar() {
  const { isAuthenticated, loading } = useAuth();

  if (loading || !isAuthenticated) return null;

  const handleLogout = async () => {
    logout();
  };

  return (
    <header className="w-full bg-blue-900 text-white flex justify-between items-center">
      <nav className="navbar">
        <div className="navbar-title">🎵 Taking Notes</div>
        <button onClick={handleLogout} className="navbar-button">Logout</button>
      </nav>
    </header>
  );
}
// src/app/Navbar.tsx

'use client';

import Link from 'next/link';
import '../styles/Navbar.css';
import { logout } from '@/utils/cognito';
import { useAuth } from '@/context/AuthContext';
import { usePathname } from 'next/navigation';
import { FaHome, FaMusic, FaBook } from 'react-icons/fa';

export default function Navbar() {
  const { isAuthenticated, loading } = useAuth();
  const pathname = usePathname();

  if (loading || !isAuthenticated) return null;

  const handleLogout = async () => {
    logout();
  };

  return (
    <header className="w-full bg-blue-900 text-white flex justify-between items-center sticky-top">
      <nav className="navbar">
        <div className="navbar-title">🎵 Taking Notes</div>

        <div className="navbar-links">
          <Link href="/home" className={pathname === '/home' ? 'active' : ''}> <FaHome /> Home </Link>
          {/* todo maybe delete */}
          <Link href="/Notes" className={pathname === '/Notes' ? 'active' : ''}> <FaMusic /> Notes </Link>
          <Link href="/MyLibrary" className={pathname === '/MyLibrary' ? 'active' : ''}> <FaBook /> My Library </Link>
        </div>

        <button onClick={handleLogout} className="navbar-button">Logout</button>
      </nav>
    </header>
  );
}
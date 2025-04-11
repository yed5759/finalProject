'use client';

import { useRouter } from 'next/navigation';
import '../styles/Navbar.css';

export default function Navbar() {
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem('accessToken');
    router.push('/landing'); // או כל עמוד אחר שתרצה להפנות אליו
  };

  return (
    <header className="w-full bg-blue-900 text-white flex justify-between items-center">
      <nav className="navbar">  {/* השתמש במחלקת ה־CSS 'navbar' */}
        <div className="navbar-title">🎵 Taking Notes</div> {/* השתמש במחלקת ה־CSS 'navbar-title' */}
        <button onClick={handleLogout} className="navbar-button">Logout</button> {/* השתמש במחלקת ה־CSS 'navbar-button' */}
      </nav>
    </header>
  );
}
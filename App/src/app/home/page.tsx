'use client';

import { useEffect } from 'react';
import { useSearchParams } from 'next/navigation';

export default function HomePage() {
  const searchParams = useSearchParams();
  const code = searchParams.get('code');

  useEffect(() => {
    console.log('🔵 [home] code from URL =', code);
    if (code) {
      // נניח שבשלב זה נשלוף טוקן כלשהו
      localStorage.setItem('accessToken', 'mock_token'); // החלפה אמיתית בהמשך
      console.log('🧩 [home] saved mock accessToken');
    }
  }, [code]);

  return (
    <div>
      <h1>Home Page</h1>
      <p>code from URL: {code}</p>
    </div>
  );
}

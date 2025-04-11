'use client';

import React, { useRef, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";
// import Navbar from '@/components/Navbar';

export default function HomePage() {
  const searchParams = useSearchParams();
  const code = searchParams.get('code');
  const [url, setUrl] = useState('');

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    console.log('🔵 [home] code from URL =', code);
    if (code) {
      // נניח שבשלב זה נשלוף טוקן כלשהו
      localStorage.setItem('accessToken', 'mock_token'); // החלפה אמיתית בהמשך
      console.log('🧩 [home] saved mock accessToken');
    }
  }, [code]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files ? event.target.files[0] : null;
    if (file) {
      console.log('קובץ שנבחר:', file);
    }
  };

  const handleSubmitUrl = () => {
    // תוסיף כאן את הלוגיקה לשליחת ה-URL
    console.log('URL שהוזן:', url);
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click(); // מבצע קליק על ה-input של הקובץ
    } else {
      console.warn("⚠️ fileInput not found in the DOM");
    }
  };

  return (
    <main
      className="w-50"
      style={{ display: 'flex', justifyContent: 'center' }}
    >
      <div>
        <h1>Home Page</h1>
        <p>code from URL: {code}</p>
      </div>
      {/* אזור העלאת קובץ */}
      <div className="border border-black">
        <h2
          className="text-xl font-semibold text-right"
          style={{ display: 'flex', justifyContent: 'center' }}
        >
          Upload File
        </h2>
        <p
          className="p-4 text-gray-600 text-right"
          style={{ display: 'flex', justifyContent: 'center' }}
        >
          Choose a file from your device to upload
        </p>
        <div className="flex justify-end p-2"
          style={{ display: 'flex', justifyContent: 'center' }}
        >
          <Button
            onClick={handleUploadClick}
            className="gap-2"
            style={{ display: 'flex', justifyContent: 'center' }}
          >
            <Upload
              className="w-4 h-4"
            />
            Upload Audio File
          </Button>

          {/* <input
              type="file"
              id="fileInput"
              className="hidden"
              onChange={handleFileUpload}
            /> */}
        </div>
      </div>

      <br></br>

      {/* אזור הזנת URL */}
      <div className="border border-black ">
        <h2 className="text-xl font-semibold text-right"
          style={{ display: 'flex', justifyContent: 'center' }}
        >Enter URL</h2>
        <p className="p-4 text-gray-600"
          style={{ display: 'flex', justifyContent: 'center' }}
        >
          Copy and paste the URL for the content in the format: https://www.example.com/video
        </p>

        <div className="space-y-4 p-1"
          style={{ display: 'flex', justifyContent: 'center' }}>
          <Input
            type="text"
            placeholder="Enter URL here"
            value={url}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)}
            dir="rtl"
          />
          <div className="flex justify-end">
            <Button onClick={handleSubmitUrl}>I've Picked a Video!!</Button>
          </div>
        </div>
      </div>
    </main>
  );
}
// src/app/landing/home.tsx

'use client';

import React, { useRef, useEffect, useState } from 'react';
// import { useSearchParams } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";
import { parseCookies } from 'nookies';
// import Navbar from '@/components/Navbar';


export default function HomePage() {
  // const searchParams = useSearchParams();
  useEffect(() => {
    // רק בצד הלקוח יש גישה ל-document
    console.log('🟢 [home.tsx] document.cookie:', document.cookie);
  }, []);
  const [url, setUrl] = useState('');

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Check if there are any tokens in the cookies
    const cookies = parseCookies();
    const accessToken = cookies.accessToken;
    console.log('🟢 [home.tsx] Cookies:', cookies);

    if (!accessToken) {
      // If no token is found, redirect to the landing page
      console.log('🟡 [home.tsx] No accessToken, redirecting to /landing');
      window.location.href = '/landing';
    } else {
      // todo If a token is found, you can send it to the server if needed
      console.log('✅ Token found in cookies:', accessToken);
    }
  }, []);
  // const code = searchParams.get("code")
  //   if (code) {
  //     // Send code to Flask
  //     fetch("http://localhost:5000/auth/callback", {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json"
  //       },
  //       body: JSON.stringify({ code })
  //     })
  //       .then(res => res.json())
  //       .then(data => {
  //         console.log("✅ Tokens from Flask:", data)
  //         // You can store tokens in cookie or context here
  //       })
  //       .catch(err => {
  //         console.error("❌ Error sending code to Flask:", err)
  //       })
  //   }
  // }, [searchParams])


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
          {/* פנייה לשרת להוסיף */}
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
            {/* פנייה לשרת להוסיף */}
          </div>
        </div>
      </div>
    </main>
  );
}
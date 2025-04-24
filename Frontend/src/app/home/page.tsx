// src/app/landing/home.tsx

'use client';

import React, { useRef, useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";

export default function HomePage() {
  const [url, setUrl] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files ? event.target.files[0] : null;
    if (file) {
      console.log('Selected file:', file);
    }
  };

  const handleSubmitUrl = () => {
    // todo Add logic for submitting the URL here
    console.log('URL שהוזן:', url);
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click(); // Triggers click on the file input
    } else {
      console.warn("⚠️ fileInput not found in the DOM");
    }
  };

  // todo notice 3 todos in the return even though he is green
  return (
    <main
      className="w-50"
      style={{ display: 'flex', justifyContent: 'center' }}
    >
      {/* File upload area */}
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
        </div>
      </div>

      {/* todo probably delete */}
      <br></br>

      {/* URL input area */}
      <div className="border border-black ">
        <h2 className="text-xl font-semibold text-right"
          style={{ display: 'flex', justifyContent: 'center' }}
        >Enter URL</h2>
        <p className="p-4 text-gray-600"
          style={{ display: 'flex', justifyContent: 'center' }}
        >
          Copy and paste the URL for the content in the format: https://www.example.com/video
          {/* todo Add server request */}
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
            {/* todo Add server request */}
          </div>
        </div>
      </div>
    </main>
  );
}
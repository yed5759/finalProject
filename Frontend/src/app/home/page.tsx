// src/app/home/page.tsx

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useRouter } from "next/navigation";
import { fetchWithRefresh } from '@/utils/cognito';

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);
  const instrumentRef = useRef<HTMLSelectElement>(null);
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (urlInputRef.current) urlInputRef.current.value = "";
    if (instrumentRef.current) instrumentRef.current.selectedIndex = 0;
  });

  async function SmartSubmit(e: { preventDefault: () => void }) {
    e.preventDefault();
    setLoading(true);
    const fileInput = document.getElementById('upload') as HTMLInputElement | null;
    const urlInput = document.getElementById('url') as HTMLInputElement | null;

    const file = fileInput?.files?.[0];
    const url = urlInput?.value.trim();

    const formData = new FormData();

    let api = "http://localhost:5000/home";

    if (file) {
      formData.append("file", file);
    } else if (url) {
      formData.append("url", url);
    } else {
      alert('Please upload a file or enter a URL!');
      setLoading(false);
      return
    }
    try {
      const response = await fetchWithRefresh(api, {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        const safeKey = encodeURIComponent(data.redirect.split("=")[1]);
        localStorage.setItem(`notes-${safeKey}`, JSON.stringify(data.notes));
        router.push(data.redirect)
      } else {
        const text = await response.text();
        alert("Server error: " + text);
        setLoading(false);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong!");
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="d-flex justify-content-center align-items-center ps-1" style={{ "marginTop": "25vh" }}>
        <div className="text-center">
          <h2 className="fw-bold">Turn your music into sheet notes</h2>
          <p className="lead"> upload a file or paste a link to get started!</p>
        </div>
      </div>
      <div className="d-flex">
        <div className="container text-center border border-1 border-black border-opacity-25 h-auto" style={{ marginLeft: '3.5rem' }}>
          <h3 className="font-monospace m-3">Upload music file</h3>
          <form className="d-flex flex-column align-items-center mb-3" id="file-form" autoComplete="off">
            <label htmlFor="upload">please choose a song</label>
            <input type="file" className="form-control w-75" id="upload" name="file" autoComplete="off" ref={fileInputRef} />
          </form>
        </div>
        <div className="container text-center border border-1 border-black border-opacity-25 h-auto" style={{ marginRight: '3.5rem' }}>
          <h3 className="font-monospace m-3">Upload URL of a song</h3>
          <form className="d-flex flex-column align-items-center mb-3" id="URL-form" autoComplete="off">
            <label htmlFor="url">Please enter the URL for the song you picked:</label>
            <input type="url" className="form-control w-75" id="url" name="url" autoComplete="off" ref={urlInputRef}
              placeholder="https://www.youtube.com/watch?v=fake1234abcd" />
          </form>
        </div>
      </div>
      <div className="d-flex justify-content-center align-items-center gap-3 m-3">
        <div className="btn-group" role="group">
          <button className="" onClick={SmartSubmit}>
            {loading && (
              <span
                className="spinner-border spinner-border-sm me-2"
                aria-hidden="true"
              ></span>
            )}
            {loading ? "Generating..." : "Generate Notes"}
          </button>
        </div>
      </div>
    </div>
  );
}

// src/app/Home/page.tsx

'use client';

import React, { useEffect, useRef } from 'react';

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);
  const instrumentRef = useRef<HTMLSelectElement>(null);

  useEffect(() => {
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (urlInputRef.current) urlInputRef.current.value = "";
    if (instrumentRef.current) instrumentRef.current.selectedIndex = 0;
  });

  async function SmartSubmit(e: { preventDefault: () => void }) {
    e.preventDefault();

    const fileInput = document.getElementById('upload') as HTMLInputElement | null;
    const urlInput = document.getElementById('url') as HTMLInputElement | null;
    const instrumentSelect = document.getElementById('instrument') as HTMLInputElement | null;

    const file = fileInput?.files?.[0];
    const url = urlInput?.value.trim();
    const instrument = instrumentSelect?.value || "piano";

    const formData = new FormData();
    formData.append("instrument", instrument);

    let api = "/Home";

    if (file) {
      formData.append("file", file);
    } else if (url) {
      formData.append("url", url);
    } else {
      alert('Please upload a file or enter a URL!');
      return
    }
    try {
      const response = await fetch(api, {
        method: 'POST',
        body: formData,
      })
      if (response.redirected) {
        window.location.href = response.url;
      } else {
        const data = await response.text();
        alert("Server response: " + data);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong!");
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
            Generate Notes
          </button>
          <select className="" id="instrument" name="format" style={{ width: '50px', borderTopLeftRadius: 0, borderBottomLeftRadius: 0 }} ref={instrumentRef}>
            <option value="piano">🎹</option>
            <option value="guitar">🎸</option>
            <option value="violin">🎻</option>
            <option value="flute">🪈</option>
          </select>
        </div>
      </div>
    </div>
  );
}

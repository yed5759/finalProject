// src/app/home/page.tsx

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useRouter } from "next/navigation";
import { isAuthenticated } from "@/utils/cognito";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";

export default function HomePage() {
  // const [url, setUrl] = useState('');
  const router = useRouter();

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);
  const instrumentRef = useRef<HTMLSelectElement>(null);

  useEffect(() => {
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (urlInputRef.current) urlInputRef.current.value = "";
    if (instrumentRef.current) instrumentRef.current.selectedIndex = 0;
  });

  // const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
  //   const file = event.target.files ? event.target.files[0] : null;
  //   if (file) {
  //     console.log('Selected file:', file);
  //   }
  // };

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

    let api = "/home";

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


  // const handleSubmitUrl = () => {
  //   // todo Add logic for submitting the URL here
  //   console.log('URL שהוזן:', url);
  // };

  // const handleUploadClick = () => {
  //   if (fileInputRef.current) {
  //     fileInputRef.current.click(); // Triggers click on the file input
  //   } else {
  //     console.warn("⚠️ fileInput not found in the DOM");
  //   }
  // };

  return (
    <div>
      <div className="d-flex justify-content-center align-items-center ps-1" style={{ "marginTop": "25vh" }}>
        <div className="text-center">
          <h2 className="fw-bold">Turn your music into sheet notes</h2>
          <p className="lead"> upload a file or paste a link to get started!</p>
        </div>
      </div>
      <div className="d-flex">
        <div className="container text-center border border-1 border-black border-opacity-25 h-auto">
          <h3 className="font-monospace m-3">Upload music file</h3>
          <form className="d-flex flex-column align-items-center mb-3" id="file-form" autoComplete="off">
            <label htmlFor="upload">please choose a song</label>
            <input type="file" className="form-control w-75" id="upload" name="file" autoComplete="off" ref={fileInputRef} />
          </form>
        </div>
        <div className="container text-center border border-1 border-black border-opacity-25 h-auto">
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

//   // todo notice 3 todos in the return even though he is green
//   return (
//     <main
//       className="w-50"
//       style={{ display: 'flex', justifyContent: 'center' }}
//     >
//       {/* File upload area */}
//       <div className="border border-black">
//         <h2
//           className="text-xl font-semibold text-right"
//           style={{ display: 'flex', justifyContent: 'center' }}
//         >
//           Upload File
//         </h2>
//         <p
//           className="p-4 text-gray-600 text-right"
//           style={{ display: 'flex', justifyContent: 'center' }}
//         >
//           Choose a file from your device to upload
//         </p>
//         <div className="flex justify-end p-2"
//           style={{ display: 'flex', justifyContent: 'center' }}
//         >
//           <Button
//             onClick={handleUploadClick}
//             className="gap-2"
//             style={{ display: 'flex', justifyContent: 'center' }}
//           >
//             <Upload
//               className="w-4 h-4"
//             />
//             Upload Audio File
//           </Button>
//         </div>
//       </div>

//       {/* todo probably delete */}
//       <br></br>

//       {/* URL input area */}
//       <div className="border border-black ">
//         <h2 className="text-xl font-semibold text-right"
//           style={{ display: 'flex', justifyContent: 'center' }}
//         >Enter URL</h2>
//         <p className="p-4 text-gray-600"
//           style={{ display: 'flex', justifyContent: 'center' }}
//         >
//           Copy and paste the URL for the content in the format: https://www.example.com/video
//           {/* todo Add server request */}
//         </p>

//         <div className="space-y-4 p-1"
//           style={{ display: 'flex', justifyContent: 'center' }}>
//           <Input
//             type="text"
//             placeholder="Enter URL here"
//             value={url}
//             onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)}
//             dir="rtl"
//           />
//           <div className="flex justify-end">
//             <Button onClick={handleSubmitUrl}>I've Picked a Video!!</Button>
//             {/* todo Add server request */}
//           </div>
//         </div>
//       </div>
//     </main>
//   );
// }
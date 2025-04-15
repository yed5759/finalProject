'use client';

import React, {useEffect} from 'react';
import {useSearchParams} from 'next/navigation';

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

    function SmartSubmit(e: { preventDefault: () => void }) {
        e.preventDefault();

        const fileInput = document.getElementById('upload') as HTMLInputElement | null;
        const urlInput = document.getElementById('url') as HTMLInputElement | null;
        const fileForm = document.getElementById('file-form') as HTMLFormElement | null;
        const urlForm = document.getElementById('URL-form') as HTMLFormElement | null;

        const file = fileInput?.files?.[0];
        const url = urlInput?.value.trim();

        if (file && fileForm) {
            fileForm.submit();
        } else if (url && urlForm) {
            urlForm.submit();
        } else {
            alert('Please upload a file or enter a URL!');
        }
    }

    return (
        <div>
            <div className="d-flex justify-content-center align-items-center ps-1" style={{"margin-top": "25vh"}}>
                <div className="text-center">
                    <h2 className="fw-bold">Turn your music into sheet notes</h2>
                    <p className="lead"> upload a file or paste a link to get started!</p>
                </div>
            </div>
            <div className="d-flex">
                <div className="container text-center border border-1 border-black border-opacity-25 h-auto">
                    <h3 className="font-monospace m-3">Upload music file</h3>
                    <form className="d-flex flex-column align-items-center mb-3" id="file-form">
                        <label htmlFor="upload">please choose a song</label>
                        <input type="file" className="form-control w-75" id="upload" name="file"/>
                    </form>
                </div>
                <div className="container text-center border border-1 border-black border-opacity-25 h-auto">
                    <h3 className="font-monospace m-3">Upload URL of a song</h3>
                    <form className="d-flex flex-column align-items-center mb-3" id="URL-form">
                        <label htmlFor="url">Please enter the URL for the song you picked:</label>
                        <input type="url" className="form-control w-75" id="url" name="url"
                               placeholder="https://www.youtube.com/watch?v=fake1234abcd"/>
                    </form>
                </div>
            </div>
            <div className="d-flex justify-content-center align-items-center gap-3 m-3">
                <div className="btn-group" role="group">
                    <button className="" onClick={SmartSubmit}>
                        Generate Notes
                    </button>
                    <select className="" id="formatSelect" name="format" style={{ width: '50px', borderTopLeftRadius: 0, borderBottomLeftRadius: 0 }}>
                        <option value="pdf">🎹</option>
                        <option value="lilypond">🎸</option>
                        <option value="musicxml">🎻</option>
                        <option value="flute">🪈</option>
                    </select>
                </div>
            </div>
        </div>
    );
}
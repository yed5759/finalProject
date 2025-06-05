// src/app/notes/page.tsx

'use client';

import { useState } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter } from 'vexflow';
import '../../styles/notes.css'
import CustomModal from '../../components/modal'

const bc = new BroadcastChannel("songs"); // ערוץ תקשורת בין טאבים

export default function Notes() {
    const [loading, setLoading] = useState(false);

    const handleAddConstSong = async () => {
        setLoading(true);
        try {
            const res = await fetch("http://localhost:5000/songs/add-const", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
            });

            if (!res.ok) throw new Error("Failed to add test song");

            // ✅ שליחת הודעה לערוץ התקשורת
            bc.postMessage({ type: "song-added" });

            alert("🎵 שיר נוסף בהצלחה!");
        } catch (error) {
            alert("שגיאה בהוספת שיר: " + (error instanceof Error ? error.message : ''));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container d-flex flex-column justify-content-center align-items-center text-center pt-5">
            <h1 className="title"><big>Taking Notes!</big></h1>
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-4">
                <button type="button" className="btn" style={{ width: '10pc', background: "#d59efb" }} data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes</button>
                <button className="btn" style={{ width: '10pc', background: "#5ac9d6" }}>Edit Notes</button>
                <button className="btn" style={{ width: '10pc', background: "#59cf59" }}>Download</button>
                {/* כפתור חדש להוספת שיר */}
                <button
                    className="btn"
                    style={{ width: '10pc', background: "#28a745", color: 'white' }}
                    onClick={handleAddConstSong}
                    disabled={loading}
                >
                    {loading ? "Adding..." : "Add Const Song"}
                </button>
            </div>
            <CustomModal />
        </div>
    );
}
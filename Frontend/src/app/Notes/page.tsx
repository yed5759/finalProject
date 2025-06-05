// src/app/notes/page.tsx

'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef, useState} from "react";

type NotesProps = {
    songName?: string;
    notes?: string[];
};
const bc = new BroadcastChannel("songs");

export default function Notes({songName, notes}: NotesProps) {
    const vfRef = useRef<HTMLDivElement>(null);
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

    // Helper to format Librosa-style notes to VexFlow keys
    const formatNote = (note: string): string =>
        note.toLowerCase().replace('♯', '#').replace(/^([a-g])([#b]?)(\d)$/i, '$1$2/$3');

    useEffect(() => {
        if (!vfRef.current) return;
        vfRef.current.innerHTML = '';
        const formatNote = (note: string): string =>
            note.toLowerCase().replace('♯', '#').replace(/^([a-g])([#b]?)(\d)$/i, '$1$2/$3');

        const givenNotes =
            notes?.map((note) =>
                new StaveNote({
                    keys: [formatNote(note)],
                    duration: 'q',
                })
            ) || [];

        const measuresPerStave = 4;
        const notesPerMeasure = 4;
        const chunkSize = measuresPerStave * notesPerMeasure;
        const groups = chunk(givenNotes, chunkSize);

        while (groups.length < 4) {
            groups.push([]); // Add empty groups (measures with no notes)
        }

        const renderer = new Renderer(vfRef.current, Renderer.Backends.SVG);
        const STAVE_HEIGHT = 100;
        const TOP_MARGIN = 20;
        const BOTTOM_BUFFER = 20;

        const height = groups.length * STAVE_HEIGHT + TOP_MARGIN + BOTTOM_BUFFER;
        renderer.resize(1400, height);

        const ctx = renderer.getContext();

        let y = 20; // vertical position for each stave
        groups.forEach(group => {
            const stave = new Stave(10, y, 1400);
            stave.addClef('treble').addTimeSignature('4/4').setContext(ctx).draw();

            if (group.length > 0) {
                // @ts-ignore
                const voice = new Voice({num_beats: chunkSize, beat_value: 4}).addTickables(group);
                new Formatter().joinVoices([voice]).format([voice], 580);
                voice.draw(ctx, stave);
            }

            y += STAVE_HEIGHT; // move down for the next stave
        });
    }, [songName, notes]);

// Utility to split notes into chunks of size n
    function chunk<T>(arr: T[], size: number): T[][] {
        return Array.from({length: Math.ceil(arr.length / size)}, (_, i) =>
            arr.slice(i * size, i * size + size)
        );
    }

    return (
        <div className="container d-flex flex-column justify-content-start align-items-center text-center"
             style={{ height: '100vh', overflow: 'hidden' }}>
            {!songName
                ? <h1 className="title"><big>Taking Notes!</big></h1>
                : <h1 className="title"><big>Taking Notes: {songName}</big></h1>
            }
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-4">
                <button type="button" className="btn" style={{width: '10pc', background: "#d59efb"}}
                        data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes
                </button>
                <button className="btn" style={{width: '10pc', background: "#5ac9d6"}}>Edit Notes</button>
                <button className="btn" style={{width: '10pc', background: "#59cf59"}}>Download</button>
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
            <div
                className="w-100 mt-4"
                style={{
                    flexGrow: 1,
                    overflowY: 'auto',
                    maxHeight: 'calc(100vh - 320px)',
                    padding: '0',
                    scrollbarWidth: 'none', // Firefox
                    msOverflowStyle: 'none'}}>
                <div ref={vfRef} style={{ width: '100%' }} />
            </div>
            <CustomModal/>
        </div>
    );
}
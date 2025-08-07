// src/app/notes/page.tsx

'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import Flow from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef, useState} from "react";
import {useSearchParams} from "next/navigation";


export default function Notes() {
    const vfRef = useRef<HTMLDivElement>(null);
    const [notes, setNotes] = useState<{ keys: string[]; duration: string }[]>([]);
    const searchParams = useSearchParams();
    const songName = searchParams.get("songName");

    useEffect(() => {
        const storedNotes = localStorage.getItem("notes");
        if (storedNotes) {
            setNotes(JSON.parse(storedNotes));
        }
    }, []);

    useEffect(() => {
        if (!vfRef.current) return;
        vfRef.current.innerHTML = '';

        const TICKS_PER_MEASURE = 4 * Flow.RESOLUTION;
        const givenNotes =
            (notes || [])
                .filter(note => {
                    return (
                        Array.isArray(note.keys) &&
                        typeof note.duration === "string" &&
                        note.keys.length > 0
                    );
                })
                .map(note => {
                    try {
                        return new StaveNote({
                            keys: note.keys,
                            duration: note.duration,
                        });
                    } catch (e) {
                        console.warn("Invalid note skipped:", note, e);
                        return null;
                    }
                })
                .filter((n): n is StaveNote => n !== null);

        const groups = chunk(givenNotes, 16); // draw 16 notes per stave

        const renderer = new Renderer(vfRef.current, Renderer.Backends.SVG);
        const ctx = renderer.getContext();

        const STAVE_HEIGHT = 100;
        const TOP_MARGIN = 10;
        const BOTTOM_BUFFER = 10;

        const height = groups.length * STAVE_HEIGHT + TOP_MARGIN + BOTTOM_BUFFER;
        renderer.resize(1400, height);

        let y = 10;

        groups.forEach((group, index) => {
            const stave = new Stave(10, y, 1400);
            stave.addClef('treble').setContext(ctx).draw();

            // Apply stave to every note (this is required for positioning)
            group.forEach(note => note.setStave(stave));

            try {
                const voice = new Voice({ time: "4/4" }).setStrict(false);
                voice.addTickables(group);
                new Formatter().joinVoices([voice]).format([voice], 1200);
                voice.draw(ctx, stave);
            } catch (err) {
                console.error(`Failed to draw voice for group ${index}:`, err);
            }

            y += STAVE_HEIGHT;
        });

        console.log("Successfully rendered", groups.length, "staves.");
    }, [songName, notes]);

    function chunk<T>(arr: T[], size: number): T[][] {
        return Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
            arr.slice(i * size, i * size + size)
        );
    }


    // @ts-ignore
    return (
        <div className="container d-flex flex-column justify-content-start align-items-center text-center"
             style={{ height: '100vh', overflow: 'hidden' }}>
            {!songName
                ? <h6 className="title"><big>Taking Notes!</big></h6>
                : <h6 className="title">{songName}</h6>
            }
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-4">
                <button type="button" className="btn" style={{width: '10pc', background: "#d59efb"}}
                        data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes
                </button>
                <button className="btn" style={{width: '10pc', background: "#5ac9d6"}}>Edit Notes</button>
                <button className="btn" style={{width: '10pc', background: "#59cf59"}}>Download</button>
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
            <CustomModal notes={notes}/>
        </div>
    );
}
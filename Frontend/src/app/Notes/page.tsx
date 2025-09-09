// src/app/notes/page.tsx

'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef, useState} from "react";
import {useSearchParams, useRouter} from "next/navigation";


export default function Notes() {
    const vfRef = useRef<HTMLDivElement>(null);
    const [notes, setNotes] = useState<{ keys: string[]; duration: string }[]>([]);
    const searchParams = useSearchParams();
    const songName = searchParams.get("songName");
    const router = useRouter();

    useEffect(() => {
        if (songName != null) {
            const storedNotes = localStorage.getItem(`notes-${songName}`);
            if (storedNotes) {
                setNotes(JSON.parse(storedNotes));
            } else {
                setNotes([]);
            }
        } else {
            setNotes([]);
        }
    }, [songName]);

    useEffect(() => {
        if (!vfRef.current) return;
        vfRef.current.innerHTML = '';

        const givenNotes = (notes || []).filter(note => {
                    return (
                        Array.isArray(note.keys) &&
                        note.keys.length > 0
                    );
                }).map(note => {
                    try {
                        return new StaveNote({
                            keys: note.keys,
                            duration: note.duration,
                        });
                    } catch (e) {
                        console.warn("Invalid note skipped:", note, e);
                        return null;
                    }
                }).filter((n): n is StaveNote => n !== null);

        const groups = chunk(givenNotes, 16); // draw 16 notes per stave

        const minStaves = 4;
        while (groups.length < minStaves) {
            groups.push([]); // empty group → renders as blank stave
        }

        const renderer = new Renderer(vfRef.current, Renderer.Backends.SVG);
        const ctx = renderer.getContext();

        const STAVE_HEIGHT = 90;
        const TOP_MARGIN = 10;
        const BOTTOM_BUFFER = 10;

        const height = groups.length * STAVE_HEIGHT + TOP_MARGIN + BOTTOM_BUFFER;
        renderer.resize(1400, height);

        let y = 10;

        groups.forEach((group, index) => {
            const stave = new Stave(10, y, 1400);
            stave.addClef('treble') .addTimeSignature('4/4').setContext(ctx).draw();

            // Apply stave to every note (this is required for positioning)
            if (group.length > 0) {
                // Attach stave + context to every note (v4 needs this)
                group.forEach(note => {
                    note.setStave(stave);
                    note.setContext(ctx);
                });

                try {
                    const voice = new Voice({numBeats: 4, beatValue: 4})
                    voice.setMode(Voice.Mode.SOFT);

                    voice.addTickables(group);
                    new Formatter().joinVoices([voice]).format([voice], 1200);
                    voice.draw(ctx, stave);
                } catch (err) {
                    console.error(`Failed to draw voice for group ${index}:`, err);
                }
            }
            y += STAVE_HEIGHT;});
    }, [songName, notes]);

    function chunk<T>(arr: T[], size: number): T[][] {
        return Array.from({length: Math.ceil(arr.length / size)}, (_, i) =>
            arr.slice(i * size, i * size + size)
        );
    }

    const handleBack = () => {
        const stack = JSON.parse(sessionStorage.getItem("navStack") || "[]");
        if (stack.length > 1) {
            // remove current page
            stack.pop();
            const last = stack.pop(); // get previous
            sessionStorage.setItem("navStack", JSON.stringify(stack));

            if (last) {
                router.push(last);
                return;
            }
        }
        // fallback
        router.push("/");
    };
    // @ts-ignore
    return (
        <div className="container d-flex flex-column justify-content-start align-items-center text-center"
             style={{height: '100vh', overflow: 'hidden'}}>
            {!songName
                ? <div className="title">Taking Notes!</div>
                : <div className="title">{songName}</div>
            }
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-2">
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
                    msOverflowStyle: 'none'
                }}>
                <div ref={vfRef} style={{width: '100%'}}/>
            </div>
            <CustomModal notes={notes}/>
            <div className="container-md justify-content-start mt-3">
                <div className="d-grid gap-2 col-6 mx-1">
                    <button type="button" className="btn btn-light rounded-0 btn-outline-dark"
                            style={{backgroundColor: "lightgray", color: "black"}}
                            onClick={handleBack}> back
                    </button>
                </div>
            </div>
        </div>
    );
}
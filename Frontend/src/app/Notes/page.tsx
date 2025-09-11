// src/app/notes/page.tsx

'use client';

import {Formatter, Renderer, Stave, StaveNote, Voice} from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef, useState} from "react";
import {useRouter, useSearchParams} from "next/navigation";
import DownloadDropdown from "../../components/dropdownSelect";


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

        const measures = groupByMeasures(givenNotes, 4);  // draw 16 notes per stave

        function groupMeasuresIntoSystems(measures: StaveNote[][], measuresPerSystem = 4) {
            const systems: StaveNote[][][] = [];
            for (let i = 0; i < measures.length; i += measuresPerSystem) {
                systems.push(measures.slice(i, i + measuresPerSystem));
            }
            return systems;
        }

        const systems = groupMeasuresIntoSystems(measures, 4);
        // --- Renderer ---
        const renderer = new Renderer(vfRef.current, Renderer.Backends.SVG);
        const ctx = renderer.getContext();

        const STAVE_HEIGHT = 90;
        const TOP_MARGIN = 0;
        const BOTTOM_BUFFER = 40;

        const height = systems.length * STAVE_HEIGHT + TOP_MARGIN + BOTTOM_BUFFER;
        renderer.resize(1400, height);

        let y = -20;

// --- ציור ---
        systems.forEach((system, index) => {
            const stave = new Stave(10, y, 1400);
            stave.addClef('treble').addTimeSignature('4/4').setContext(ctx).draw();

            // מאחדים את כל התווים מהתיבות
            const notesInSystem = system.flat();

            if (notesInSystem.length > 0) {
                notesInSystem.forEach(note => {
                    note.setStave(stave);
                    note.setContext(ctx);
                });

                try {
                    const voice = new Voice({ numBeats: 4, beatValue: 4 });
                    voice.setMode(Voice.Mode.SOFT);

                    voice.addTickables(notesInSystem);
                    new Formatter().joinVoices([voice]).format([voice], 1200);
                    voice.draw(ctx, stave);
                } catch (err) {
                    console.error(`Failed to draw voice for system ${index}:`, err);
                }
            }

            y += STAVE_HEIGHT;
        });
    }, [songName, notes]);


    function groupByMeasures(notes: StaveNote[], beatsPerMeasure = 4): StaveNote[][] {
        const measures: StaveNote[][] = [];
        let current: StaveNote[] = [];
        let beats = 0;

        notes.forEach(note => {
            current.push(note);
            beats += durationToBeats(note.getDuration());

            if (beats >= beatsPerMeasure) {
                measures.push(current);
                current = [];
                beats = 0;
            }
        });

        if (current.length > 0) {
            measures.push(current);
        }

        return measures;
    }

    function durationToBeats(duration: string): number {
        switch (duration) {
            case "w": return 4;     // whole note
            case "h": return 2;     // half note
            case "q": return 1;     // quarter note
            case "8": return 0.5;   // eighth note
            case "16": return 0.25; // sixteenth note
            default: return 1;      // ברירת מחדל לרבע
        }
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

            <div className="d-flex gap-3 mt-3">
                <button type="button" className="btn" style={{width: '10pc', background: "#d59efb"}}
                        data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes
                </button>
                <button className="btn" style={{width: '10pc', background: "#5ac9d6"}}>Edit Notes</button>
                <DownloadDropdown vfRef={vfRef} notes={notes}/>
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
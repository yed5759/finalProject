// src/app/Notes/page.tsx

'use client';

import { Formatter, Renderer, Stave, StaveNote, Voice } from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import DownloadDropdown from "../../components/dropdownSelect";
import NotesEditor from '../../components/NotesEditor';
import { rawToVexflow, type RawNote, type VexNote } from '../../utils/notes';
import { buildRawFromVex } from '../../utils/notes';
import { fetchWithRefresh } from '@/utils/cognito';


export default function Notes() {
    const vfRef = useRef<HTMLDivElement>(null);

    const [notes, setNotes] = useState<{ keys: string[]; duration: string }[]>([]);

    const searchParams = useSearchParams();
    const freshParam = searchParams.get('fresh'); // '1' | null
    const songName = searchParams.get("songName");
    const id = searchParams.get("id") || null;
    const titleKey = songName ? decodeURIComponent(songName) : "";
    const router = useRouter();
    const [raw, setRaw] = useState<RawNote[]>([]);
    const [bpm, setBpm] = useState<number>(120);
    const [editOpen, setEditOpen] = useState(false);

    useEffect(() => {
        if (!titleKey) {
            setRaw([]);
            setBpm(120);
            return;
        }
        const isFresh = freshParam === '1';
        const freshTitle = sessionStorage.getItem('fresh_title');
        if (isFresh && freshTitle === titleKey) {
            // 1) boot from the freshly-generated payload
            const vex = JSON.parse(sessionStorage.getItem('fresh_vex') || '[]') as VexNote[];
            const b = Number(sessionStorage.getItem('fresh_bpm') || 120);


            setNotes(vex);
            // seed editor from what we show
            const boot = buildRawFromVex(vex, Number.isFinite(b) && b > 0 ? b : 120);
            setRaw(boot);
            setBpm(Number.isFinite(b) && b > 0 ? b : 120);

            // 2) overwrite the local cache with the fresh data
            localStorage.setItem(`notes-${songName}`, JSON.stringify(vex));
            localStorage.setItem(`raw-${titleKey}`, JSON.stringify(boot));
            localStorage.setItem(`bpm-${songName}`, String(b));

            // 3) clear the one-shot session payload so it doesn't reapply on refresh
            sessionStorage.removeItem('fresh_title');
            sessionStorage.removeItem('fresh_vex');
            sessionStorage.removeItem('fresh_bpm');
        } else {
            // fallback: use whatever is cached (edits) for this title
            const vex = JSON.parse(localStorage.getItem(`notes-${songName}`) || '[]') as VexNote[];
            const r = JSON.parse(localStorage.getItem(`raw-${songName}`) || '[]') as RawNote[];
            const b = Number(localStorage.getItem(`bpm-${songName}`) || 120);
            const boot = r.length ? r : buildRawFromVex(vex, Number.isFinite(b) && b > 0 ? b : 120);
            setNotes(vex);
            setRaw(boot);
            setBpm(Number.isFinite(b) && b > 0 ? b : 120);
            if (!vex.length) {
                setNotes([]);
            }
            if (!r.length && vex.length) {
                localStorage.setItem(`raw-${titleKey}`, JSON.stringify(boot));
            }
        }
    }, [titleKey, freshParam]);

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
                const duration = note.duration;

                return new StaveNote({
                    keys: note.keys,
                    duration: duration,
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
                    const voice = new Voice({ numBeats: 16, beatValue: 4 });
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

    const applyEdits = async () => {
        const vex = rawToVexflow(raw, bpm);   // derive VexFlow notes from editable raw
        setNotes(vex);
        if (songName) {
            localStorage.setItem(`notes-${songName}`, JSON.stringify(vex));
            localStorage.setItem(`raw-${songName}`, JSON.stringify(raw));
            localStorage.setItem(`bpm-${songName}`, String(bpm));
        }
        setEditOpen(false);
        if (id) {
            try {
                const res = await fetchWithRefresh(`http://localhost:5000/songs/${id}`, {
                    method: "PATCH",
                    headers: {
                        "Content-Type": "application/json",
                        Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                    },
                    body: JSON.stringify({
                        title: songName,
                        notes: vex
                    }),
                });
            } catch (error: any) {
                alert("שגיאה בעדכון השיר: " + (error?.message || ""));
            }
        }
    };


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

    const openEditor = () => {
        if (!raw.length && notes.length) {
            const boot = buildRawFromVex(notes, bpm);
            setRaw(boot);
            if (titleKey) localStorage.setItem(`raw-${titleKey}`, JSON.stringify(boot));
        }
        setEditOpen(true);
    };


    // @ts-ignore
    return (
        <div className="container d-flex flex-column justify-content-start align-items-center text-center"
            style={{ height: '100vh', overflow: 'hidden' }}>
            {!songName
                ? <div className="title">Taking Notes!</div>
                : <div className="title">{songName}</div>
            }
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-3">
                <button type="button" className="btn" style={{ width: '10pc', background: "#d59efb" }}
                    data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes
                </button>
                <button className="btn" style=
                    {{
                        width: '10pc',
                        background: "#5ac9d6"
                    }}
                    onClick={openEditor}>Edit Notes</button>
                <DownloadDropdown vfRef={vfRef as React.RefObject<HTMLDivElement>} notes={notes} />
            </div>
            <div
                className="w-100 mt-4"
                style={{
                    flexGrow: 1,
                    overflowY: 'auto',
                    maxHeight: 'calc(100vh - 600px)',
                    padding: '0',
                    scrollbarWidth: 'none', // Firefox
                    msOverflowStyle: 'none'
                }}>
                <div ref={vfRef} style={{ width: '100%' }} />
            </div>
            <CustomModal notes={notes.map(n => ({
                keys: n.keys,
                duration: n.duration
            }))} />
            <div className="container-md justify-content-start mt-3">
                <div className="d-grid gap-2 col-6 mx-1">
                    <button
                        type="button"
                        className="btn btn-light rounded-0 btn-outline-dark"
                        style={{
                            position: 'fixed',
                            bottom: '30px',
                            left: '80px',
                            width: '15ch',
                            backgroundColor: 'lightgray',
                            color: 'black',
                            zIndex: 1000
                        }}
                        onClick={handleBack}
                    >
                        Back
                    </button>
                </div>
            </div>
            <NotesEditor
                open={editOpen}
                onClose={() => setEditOpen(false)}
                bpm={bpm}
                setBpm={setBpm}
                rawNotes={raw}
                onChange={setRaw}
                onApply={applyEdits}
            />
        </div>
    );
}
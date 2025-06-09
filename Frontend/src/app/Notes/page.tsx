// src/app/notes/page.tsx

'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef} from "react";

type NotesProps = {
    songName?: string;
    notes?: string[];
};

export default function Notes({songName, notes}: NotesProps) {
    const vfRef = useRef<HTMLDivElement>(null);

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
        const TOP_MARGIN = 10;
        const BOTTOM_BUFFER = 10;

        const height = groups.length * STAVE_HEIGHT + TOP_MARGIN + BOTTOM_BUFFER;
        renderer.resize(1400, height + 10);

        const ctx = renderer.getContext();

        let y = 10; // vertical position for each stave
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

    // @ts-ignore
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
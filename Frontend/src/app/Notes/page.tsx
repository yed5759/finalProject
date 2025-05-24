'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import '../../styles/Notes.css'
import CustomModal from '../../components/modal'
import {useEffect, useRef} from "react";

type NotesProps = { songName?: string };

export default function Notes({ songName }: NotesProps) {
    const vfRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!vfRef.current) return;
        // Clear out any old rendering
        vfRef.current.innerHTML = '';

        // 1) Create the SVG renderer
        const renderer = new Renderer(vfRef.current, Renderer.Backends.SVG);
        renderer.resize(700, 600);
        const ctx = renderer.getContext();

        // 2) Draw a stave
        const stave = new Stave(5, 20, 680);
        stave.addClef('treble').addTimeSignature('4/4').setContext(ctx).draw();
        const notes = [
            new StaveNote({ keys: ['c/4'], duration: 'q' }),
            new StaveNote({ keys: ['d/4'], duration: 'q' }),
            new StaveNote({ keys: ['b/4'], duration: 'qr' }), // quarter rest
            new StaveNote({ keys: ['e/4'], duration: 'q' }),
        ];

        // @ts-ignore
        const voice = new Voice({ num_beats: 4, beat_value: 4 }).addTickables(notes);
        new Formatter().joinVoices([voice]).format([voice], 580);
        voice.draw(ctx, stave);
    }, []);

    return (
        <div className="container d-flex flex-column justify-content-center align-items-center text-center pt-2">
            { !songName
                ? <h1 className="title"><big>Taking Notes!</big></h1>
                : <h1 className="title"><big>Taking Notes: {songName}</big></h1>
            }
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-4">
                <button type="button" className="btn" style={{width: '10pc', background: "#d59efb"}} data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes</button>
                <button className="btn" style={{width: '10pc', background: "#5ac9d6"}}>Edit Notes</button>
                <button className="btn" style={{width: '10pc', background: "#59cf59"}}>Download</button>
            </div>
            <div ref={vfRef} className="mt-4" />
            <CustomModal />
        </div>
    );
}
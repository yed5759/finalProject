'use client';

import {Renderer, Stave, StaveNote, Voice, Formatter} from 'vexflow';
import '../styles/Notes.css'
import CustomModal from '../../components/modal'

export default function Notes() {
    return (
        <div className="container d-flex flex-column justify-content-center align-items-center text-center pt-5">
            <h1 className="title"><big>Taking Notes!</big></h1>
            <div className="underline"></div>

            <div className="d-flex gap-3 mt-4">
                <button type="button" className="btn" style={{width: '10pc', background: "#d59efb"}} data-bs-toggle="modal" data-bs-target="#staticBackdrop">Save Notes</button>
                <button className="btn" style={{width: '10pc', background: "#5ac9d6"}}>Edit Notes</button>
                <button className="btn" style={{width: '10pc', background: "#59cf59"}}>Download</button>
            </div>
            <CustomModal />
        </div>
    );
}
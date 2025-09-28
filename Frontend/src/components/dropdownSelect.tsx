// src/app/components/dropdownSelect.tsx

import React, { useState } from "react";
import jsPDF from "jspdf";
// @ts-ignore
import canvg from "canvg";
import { Midi } from "@tonejs/midi"


type Props = {
    vfRef: React.RefObject<HTMLDivElement>;
    notes: any[];
};

export default function DownloadDropdown({ vfRef, notes }: Props) {
    const [selected, setSelected] = useState<string>("(None)");

    const handleDownload = () => {
        if (selected === "(None)") {
            alert("First select a format to download.");
        } else if (selected === "pdf") {
            void downloadPDF();
        } else {
            void downloadMIDI();
        }
        setSelected("(None)");
    }

    async function downloadPDF() {
        const svg = vfRef.current?.querySelector("svg");
        if (!svg) return;

        const svgText = new XMLSerializer().serializeToString(svg);

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const v = await canvg.from(ctx, svgText);
        await v.render();

        const imgData = canvas.toDataURL("image/png");
        const pdf = new jsPDF("l", "pt", "a4");
        pdf.addImage(imgData, "PNG", 10, 10, 800, 600);
        pdf.save("sheet_music.pdf");
    }
    async function downloadMIDI() {
        const midi = new Midi();
        const track = midi.addTrack();

        const bpm = 120;
        const quarterNote = 60 / bpm; // שניות לרבע
        let currentTime = 0;

        notes.forEach((note) => {
            const durationMap: Record<string, number> = {
                "w": 4,
                "h": 2,
                "q": 1,
                "8": 0.5,
                "16": 0.25,
            };
            const beats = durationMap[note.duration] || 1;
            const noteDuration = beats * quarterNote;

            note.keys.forEach((k: string) => {
                const [letter, octaveStr] = k.split("/");
                const octave = parseInt(octaveStr, 10);
                const pitchClass = {
                    c: 0, "c#": 1, d: 2, "d#": 3, e: 4, f: 5, "f#": 6,
                    g: 7, "g#": 8, a: 9, "a#": 10, b: 11
                }[letter.toLowerCase()] ?? 0;

                const midiNumber = 12 * (octave + 1) + pitchClass;

                track.addNote({
                    midi: midiNumber,
                    time: currentTime,
                    duration: noteDuration,
                    velocity: 0.8,
                });
            });

            currentTime += noteDuration;
        });

        const bytes = midi.toArray();


        // const blob = new Blob([bytes], { type: "audio/midi" });
        const arrayBuffer = bytes.buffer instanceof ArrayBuffer
            ? bytes.buffer
            : Uint8Array.from(bytes).buffer;

        const blob = new Blob([arrayBuffer], { type: "audio/midi" });

        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "sheet_music.mid";
        a.click();
        URL.revokeObjectURL(url);
    }


    return (
        <div className="btn-group">
            {/* Main action button could use the selected value */}
            <button
                type="button"
                className="btn"
                style={{ width: "10pc", background: "#59cf59" }}
                onClick={handleDownload}
            >
                Download {selected}
            </button>

            {/* Split dropdown toggle */}
            <button
                type="button"
                className="btn dropdown-toggle dropdown-toggle-split"
                style={{ background: "#59cf59" }}
                data-bs-toggle="dropdown"
                aria-expanded="false"
            >
                <span className="visually-hidden">Toggle Dropdown</span>
            </button>

            {/* Dropdown options */}
            <ul className="dropdown-menu">
                {["(None)", "pdf", "midi"].map((opt) => (
                    <li key={opt}>
                        <button
                            className="dropdown-item"
                            onClick={() => setSelected(opt)}
                        >
                            {opt}
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
}
// src/utils/notes.ts
export type RawNote = {
    id: string;            // unique identifier
    type: 'note' | 'chord' | 'rest';
    pitches: number[];     // MIDI
    start: number;         // seconds
    duration: number;      // seconds
    velocity: number;      // 0..1
};

export type VexNote = {
    keys: string[];        // e.g., ["c/4","e/4"]
    duration: string;      // "q", "h", "8", ...
    isRest?: boolean;
};

// MIDI -> VexFlow key
const NAMES = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
export const midiToVexKey = (m: number) => `${NAMES[m % 12]}/${Math.floor(m / 12) - 1}`;

const vexKeyToMidi = (k: string) => {
    const [pc, octStr] = k.toLowerCase().split('/');
    const idx = NAMES.indexOf(pc);
    const oct = Number(octStr);
    return (oct + 1) * 12 + (idx >= 0 ? idx : 0);
};

const durationToBeatsVex = (d: string) => {
    const dotted = d.endsWith("d");
    const base = dotted ? d.slice(0, -1) : d;

    let beats =
        base === "w" ? 4 :
            base === "h" ? 2 :
                base === "q" ? 1 :
                    base === "8" ? 0.5 :
                        base === "16" ? 0.25 :
                            base === "32" ? 0.125 :
                                1; // ברירת מחדל רבע

    if (dotted) {
        beats *= 1.5;
    }

    return beats;
};

export function buildRawFromVex(vex: VexNote[], bpm: number): RawNote[] {
    const beat = 60 / bpm;
    let t = 0;
    return vex.map(v => {
        const beats = durationToBeatsVex(v.duration);
        const dur = beats * beat;
        const pitches = v.isRest ? [] : v.keys.map(vexKeyToMidi);
        const row: RawNote = {
            id: crypto.randomUUID(),
            type: v.isRest
                ? 'rest'
                : pitches.length > 1 ? 'chord' : 'note',
            pitches,
            start: t,
            duration: dur,
            velocity: v.isRest ? 0 : 0.7
        };
        t += dur;
        return row;
    });
}

export function secondsToDuration(sec: number, bpm: number): string {
    const beatSec = 60 / bpm;
    const beats = sec / beatSec;

    // טבלה בסיסית + גרסאות מנוקדות (1.5 ×)
    const base: [number, string][] = [
        [4, 'w'],
        [2, 'h'],
        [1, 'q'],
        [0.5, '8'],
        [0.25, '16'],
        [0.125, '32'],
    ];

    const table: [number, string][] = [];
    for (const [val, name] of base) {
        table.push([val, name]);             // רגיל
        table.push([val * 1.5, name + 'd']); // מנוקד
    }

    // בוחרים את הערך הכי קרוב
    let best = 'q', errMin = Infinity;
    for (const [val, name] of table) {
        const e = Math.abs(beats - val);
        if (e < errMin) {
            errMin = e;
            best = name;
        }
    }
    return best;
}

export function rawToVexflow(raw: RawNote[], bpm: number): VexNote[] {
    const out: VexNote[] = [];
    let prevEnd = 0;

    for (const n of raw.sort((a, b) => a.start - b.start)) {
        out.push({
            keys: n.pitches.map(midiToVexKey),
            duration: secondsToDuration(n.duration, bpm),
            isRest: n.type === 'rest'
        });

        prevEnd = n.start + n.duration;
    }

    return out;
}
// quantize start/duration to grid (e.g. 1/8)
export function quantizeRaw(
    notes: RawNote[],
    bpm: number,
    denom: 4 | 8 | 16 = 8
): RawNote[] {
    const beatSec = 60 / bpm;
    const grid = 4 / denom * beatSec; // length of the grid in seconds
    const q = (x: number) => Math.round(x / grid) * grid;
    return notes.map(n => ({
        ...n,
        start: Math.max(0, q(n.start)),
        duration: Math.max(grid, q(n.duration)),
    }));
}
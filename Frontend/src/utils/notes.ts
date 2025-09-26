// src/utils/notes.ts
export type RawNote = {
    type: 'note' | 'chord';
    pitches: number[];     // MIDI
    start: number;         // seconds
    duration: number;      // seconds
    velocity: number;      // 0..1
};

export type VexNote = {
    keys: string[];        // e.g., ["c/4","e/4"]
    duration: string;      // "q", "h", "8", ...
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

const durationToBeatsVex = (d: string) =>
    d === 'w' ? 4 : d === 'h' ? 2 : d === '8' ? 0.5 : d === '16' ? 0.25 : 1;

export function buildRawFromVex(vex: { keys: string[]; duration: string }[], bpm: number) {
    const beat = 60 / bpm;
    let t = 0;
    return vex.map(v => {
        const beats = durationToBeatsVex(v.duration);
        const dur = beats * beat;
        const pitches = v.keys.map(vexKeyToMidi);
        const row = {
            type: pitches.length > 1 ? 'chord' as const : 'note' as const,
            pitches, start: t, duration: dur, velocity: 0.7
        };
        t += dur;
        return row;
    });
}

// seconds -> VexFlow duration using bpm (simple snap)
export function secondsToDuration(sec: number, bpm: number): string {
    const beatSec = 60 / bpm;
    const beats = sec / beatSec;
    const table: [number, string][] = [
        [4, 'w'], [2, 'h'], [1, 'q'], [0.5, '8'], [0.25, '16'], [0.125, '32']
    ];
    let best = 'q', errMin = Infinity;
    for (const [val, name] of table) {
        const e = Math.abs(beats - val);
        if (e < errMin) { errMin = e; best = name; }
    }
    return best;
}

export function rawToVexflow(raw: RawNote[], bpm: number): VexNote[] {
    return raw.map(n => ({
        keys: n.pitches.map(midiToVexKey),
        duration: secondsToDuration(n.duration, bpm),
    }));
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
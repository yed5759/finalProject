// src/components/NotesEditor.tsx
'use client';
import { useState } from 'react';
import type { RawNote } from '../utils/notes';

type Props = {
  open: boolean;
  onClose: () => void;
  bpm: number;
  setBpm: (v: number) => void;
  rawNotes: RawNote[];
  onChange: (next: RawNote[]) => void;   // live edits
  onApply: () => void;                   // regenerate + save
};

export default function NotesEditor({
  open, onClose, bpm, setBpm, rawNotes, onChange, onApply
}: Props) {
  const [denom, setDenom] = useState<4 | 8 | 16>(8);

  if (!open) return null;

  const update = (idx: number, patch: Partial<RawNote>) => {
    const next = rawNotes.slice();
    next[idx] = { ...next[idx], ...patch };
    onChange(next);
  };

  const addRow = () => onChange([...rawNotes, {
    type: 'note', pitches: [60], start: 0, duration: 0.5, velocity: 0.7
  }]);

  const removeRow = (i: number) => onChange(rawNotes.filter((_, idx) => idx !== i));

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,.45)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
    }}>
      <div style={{
        background: '#fff', width: 'min(1000px, 95vw)', maxHeight: '85vh',
        overflow: 'auto', borderRadius: 12, padding: 16
      }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
          <h3 style={{ margin: 0 }}>Edit Notes</h3>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <label>BPM:
              <input type="number" value={bpm} min={20} max={300}
                onChange={e => setBpm(Number(e.target.value) || 120)}
                style={{ width: 80, marginLeft: 6 }} />
            </label>
            <label>Quantize:
              <select value={denom} onChange={e => setDenom(Number(e.target.value) as 4 | 8 | 16)}
                style={{ marginLeft: 6 }}>
                <option value={4}>¼</option>
                <option value={8}>⅛</option>
                <option value={16}>¹⁶</option>
              </select>
            </label>
            <button type="button" onClick={() => {
              const { quantizeRaw } = require('@/utils/notes');
              onChange(quantizeRaw(rawNotes, bpm, denom));
            }}>Apply Quantize</button>
            <button type="button" onClick={addRow}>+ Add</button>
            <button type="button" onClick={onApply} style={{ background: '#222', color: '#fff' }}>Save</button>
            <button type="button" onClick={onClose}>Close</button>
          </div>
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th>#</th><th>Type</th><th>Pitches (comma MIDI)</th>
              <th>Start (s)</th><th>Duration (s)</th><th>Vel</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rawNotes.map((n, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>
                  <select value={n.type} onChange={e => update(i, { type: e.target.value as any })}>
                    <option value="note">note</option>
                    <option value="chord">chord</option>
                  </select>
                </td>
                <td>
                  <input
                    value={n.pitches.join(',')}
                    onChange={e => {
                      const nums = e.target.value
                        .split(',')
                        .map(s => s.trim())
                        .filter(Boolean)
                        .map(x => Math.max(0, Math.min(127, Number(x) || 0)));
                      update(i, { pitches: nums.length ? nums : [60] });
                    }}
                    style={{ width: '100%' }}
                  />
                </td>
                <td>
                  <input type="number" step="0.01" value={n.start}
                    onChange={e => update(i, { start: Math.max(0, Number(e.target.value) || 0) })}
                    style={{ width: 90 }} />
                </td>
                <td>
                  <input type="number" step="0.01" value={n.duration}
                    onChange={e => update(i, { duration: Math.max(0.01, Number(e.target.value) || 0.01) })}
                    style={{ width: 90 }} />
                </td>
                <td>
                  <input type="number" step="0.01" min="0" max="1" value={n.velocity}
                    onChange={e => update(i, { velocity: Math.max(0, Math.min(1, Number(e.target.value) || 0)) })}
                    style={{ width: 70 }} />
                </td>
                <td>
                  <button type="button" onClick={() => removeRow(i)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
// src/app/myLibrary/page.tsx

"use client";

import React, { useState, useEffect } from 'react';
import { MdDelete, MdShare, MdEdit } from 'react-icons/md';
import { useRouter } from "next/navigation";
import { fetchWithRefresh } from '../../utils/cognito';

type Song = {
    id: string;
    title: string;
    artist?: string;
    notes: string[];
    tags?: string[];
};

const bc = new BroadcastChannel("songs");

export default function MyLibrary() {
    // Songs list state
    const [songs, setSongs] = useState<Song[]>([]);
    // State for search query
    const [searchQuery, setSearchQuery] = useState('');
    // State for inline editing
    const [editingSongId, setEditingSongId] = useState<string | null>(null);
    const [editingValues, setEditingValues] = useState<{ title: string; artist?: string; tags?: string[]; newTag?: string }>({ title: '', artist: '', tags: [], newTag: '' });
    const router = useRouter();

    // Filter songs based on search query
    const query = searchQuery.toLowerCase();
    const filteredSongs = songs.filter((song) =>
        song.title?.toLowerCase().includes(query) ||
        song.artist?.toLowerCase().includes(query) ||
        song.tags?.some(tag => tag?.toLowerCase().includes(query))
    );

    async function getSong(id: string, name: string) {
        try {
            const res = await fetchWithRefresh(`http://localhost:5000/songs/${id}`, {
                method: "GET",
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`
                },
            });
            if (res.ok) {
                const data = await res.json()
                const notes = data['notes']
                localStorage.setItem(`notes-${name}`, JSON.stringify(notes))
                router.push(`/Notes?songName=${name}`)
            } else if (res.status === 404) {
                alert("השיר לא נמצא – כנראה נמחק");
                setSongs(prev => prev.filter(song => song.id !== id)); // עדכון ה־state
            } else {
                throw new Error(`Unexpected response: ${res.status}`);
            }
        } catch (error: any) {
            throw new Error(error?.message || "Failed to fetch song")
        }
    }

    // Handle deleting a song width given id
    const handleDelete = async (id: string, event: React.MouseEvent) => {
        event.stopPropagation();
        try {
            const res = await fetchWithRefresh(`http://localhost:5000/songs/${id}`, {
                method: "DELETE",
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
            });
            // Delete from client side
            if (res.ok) {
                setSongs(prevSongs => prevSongs.filter(song => song.id !== id));
            }
        } catch (error) {
            console.error("Error deleting song:", error);
            alert("שגיאה במחיקת שיר");
        }
    };

    // // Handle sharing a song (this is just a placeholder)
    // const handleShare = (song: Song) => {
    //     // For now, just alert the song title and artist
    //     alert(`Sharing song: ${song.title}`);
    // };

    // Handle inline editing
    const handleEdit = (song: Song) => {
        setEditingSongId(song.id);
        setEditingValues({
            title: song.title,
            artist: song.artist || '',
            tags: song.tags || [],
        });
    };

    const handleChange = (field: keyof typeof editingValues, value: string) => {
        setEditingValues(prev => ({
            ...prev,
            [field]: field === "tags" ? value.split(",").map(t => t.trim()).filter(t => t) : value
        }));
    };
    const addTag = () => {
        const tag = (editingValues.newTag || '').trim();
        if (tag) {
            setEditingValues(prev => ({ ...prev, tags: [...(prev.tags || []), tag], newTag: '' }));
        }
    };

    const handleSaveEdit = async (songId: string) => {
        try {
            const oldSong = songs.find(s => s.id === songId);
            if (!oldSong) return;

            const res = await fetchWithRefresh(`http://localhost:5000/songs/${songId}`, {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
                body: JSON.stringify({
                    title: editingValues.title,
                    artist: editingValues.artist,
                    tags: editingValues.tags,
                }),
            });

            if (!res.ok) throw new Error("Failed to update song");

            // Update notes key if title changed
            if (oldSong.title !== editingValues.title) {
                const oldKey = `notes-${encodeURIComponent(oldSong.title)}`;
                const newKey = `notes-${encodeURIComponent(editingValues.title)}`;
                const notes = localStorage.getItem(oldKey);
                if (notes) {
                    localStorage.setItem(newKey, notes);
                    // optional: remove old key
                    // localStorage.removeItem(oldKey);
                }
            }

            setSongs(prev => prev.map(s => s.id === songId ? { ...s, ...editingValues } : s));
            setEditingSongId(null);
        } catch (error: any) {
            alert("שגיאה בעדכון השיר: " + (error?.message || ""));
        }
    };

    // ✅ הפונקציה לשליפת שירים (חשוב שתהיה נפרדת כדי שנוכל לקרוא לה מאירועים)
    const fetchSongs = async () => {
        try {

            console.log("Access Token:", localStorage.getItem("accessToken"));
            console.log("ID Token:", localStorage.getItem("id_token"));
            console.log("Refresh Token:", localStorage.getItem("refreshToken"));

            const res = await fetchWithRefresh("http://localhost:5000/songs", {
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
            });

            if (res.ok) {
                const data = await res.json();
                setSongs(data);
            }
        } catch (error) {
            console.error("Error fetching songs:", error);
            alert("Error fetching songs");
        }
    };

    useEffect(() => {
        // Get list of songs
        fetchSongs().then(() => { });
        // Listener to song-added event
        bc.onmessage = (event) => {
            if (event.data?.type === "song-added") {
                fetchSongs().then(() => { }); // Refresh list of songs
            }
        };

        return () => {
            bc.close();
        };
    }, []);

    return (
        <div className="container d-flex flex-column justify-content-start align-items-start text-start">
            <h2>My Library</h2>
            <input className="form-control"
                type="text"
                placeholder="Search for song, artist, or tag"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)} />
            <div className="w-100"
                style={{
                    maxHeight: '500px',
                    overflowY: 'scroll'
                }}>
                <div className="list-group list-group-numbered">
                    {filteredSongs.map((song, index) => (
                        <div
                            role="button"
                            style={{
                                padding: '10px',
                                borderBottom: index !== filteredSongs.length - 1 ? '1px solid black' : 'none',
                                backgroundColor: 'seashell',
                                cursor: "pointer",
                            }}
                            className="list-group-item list-group-item-action d-flex"
                            key={song.id} onClick={() => { if (editingSongId !== song.id) getSong(song.id, song.title); }}
                        >
                            {editingSongId === song.id ? (
                                <div className="flex-fill ps-3 d-flex flex-column gap-2">
                                    <div className="ps-3 mb-2">
                                        <label><strong>Title:</strong></label>
                                        <input type="text"
                                            className="form-control"
                                            value={editingValues.title}
                                            onChange={e => handleChange("title", e.target.value)}
                                            onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(song.id); }}
                                        />
                                    </div>
                                    <div className="ps-3 mb-2">
                                        <label><strong>Artist:</strong></label>
                                        <input type="text"
                                            className="form-control"
                                            value={editingValues.artist}
                                            onChange={e => handleChange("artist", e.target.value)}
                                            onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(song.id); }}
                                        />
                                    </div>
                                    <div className="ps-3 mb-2 d-flex flex-column gap-1">
                                        <label><strong>Tags:</strong></label>
                                        <div className="d-flex gap-2">
                                            <input type="text"
                                                className="form-control"
                                                placeholder="Add tag"
                                                value={editingValues.newTag || ''}
                                                onChange={e => setEditingValues(prev => ({ ...prev, newTag: e.target.value }))}
                                                onKeyDown={e => { if (e.key === 'Enter') { addTag(); e.preventDefault(); } }}
                                            />
                                            <button type="button" onClick={(e) => { e.stopPropagation(); addTag(); }} style={{ padding: '0 4px', cursor: 'pointer' }}>➕</button>
                                        </div>
                                        <div>
                                            {(editingValues.tags || []).map((tag, i) => (
                                                <span key={i} style={{ marginRight: '5px', display: 'inline-flex', alignItems: 'center', gap: '2px' }}>
                                                    #{tag}
                                                    <button type="button" onClick={(e) => {
                                                        e.stopPropagation();
                                                        setEditingValues(prev => ({ ...prev, tags: prev.tags?.filter((_, idx) => idx !== i) }));
                                                    }} style={{ padding: '0 4px', cursor: 'pointer' }}>➖</button>
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="d-flex gap-2">
                                        <button onClick={(e) => { e.stopPropagation(); handleSaveEdit(song.id); }}>✔️</button>
                                        <button onClick={(e) => { e.stopPropagation(); setEditingSongId(null); }}>❌</button>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex-fill">

                                    <h5 style={{ marginBottom: '0px' }} className="ps-3"><strong>{song.title}</strong></h5>
                                    {song.artist && (
                                        <p style={{ marginBottom: '0px' }} className="ps-3">
                                            <strong>Artist:</strong> {song.artist}</p>
                                    )}
                                    {song.tags && song.tags.length > 0 && (
                                        <p style={{ marginBottom: '0px' }} className="ps-3">
                                            <strong>Tags:</strong> {song.tags.join(', ')}</p>
                                    )}
                                </div>
                            )}

                            {/* Buttons container */}
                            <div className="d-flex justify-content-end">
                                {/* Delete icon button */}
                                <button
                                    onClick={(e) => handleDelete(song.id, e)}
                                    style={{
                                        padding: '5px',
                                        backgroundColor: 'transparent',
                                        border: 'none',
                                        cursor: 'pointer',
                                        fontSize: '20px',
                                    }}
                                    title="Delete">
                                    <MdDelete></MdDelete>
                                </button>
                                {/* todo delete */}
                                {/* Share icon button */}
                                {/* <button
                                    onClick={() => handleShare(song)}
                                    style={{
                                        padding: '5px',
                                        backgroundColor: 'transparent',
                                        border: 'none',
                                        cursor: 'pointer',
                                        fontSize: '20px',
                                    }}
                                    title="Share">
                                    <MdShare />
                                </button> */}
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleEdit(song); }}
                                    style={{
                                        padding: '5px',
                                        backgroundColor: 'transparent',
                                        border: 'none',
                                        cursor: 'pointer',
                                        fontSize: '20px',
                                    }}
                                    title="Edit">
                                    <MdEdit></MdEdit>
                                </button>
                            </div>
                            {/* Display artist only if available */}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
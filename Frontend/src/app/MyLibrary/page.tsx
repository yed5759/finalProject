// src/app/myLibrary/page.tsx

"use client";

import React, { useState, useEffect } from 'react';
import { MdDelete, MdShare } from 'react-icons/md';
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
            }
        } catch (error: any) {
            throw new Error(error?.message || "Failed to fetch song")
        }
    }

    // Handle deleting a song width given id
    const handleDelete = async (id: string) => {
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

    // Handle sharing a song (this is just a placeholder)
    const handleShare = (song: Song) => {
        // For now, just alert the song title and artist
        alert(`Sharing song: ${song.title}`);
    };

    // ✅ הפונקציה לשליפת שירים (חשוב שתהיה נפרדת כדי שנוכל לקרוא לה מאירועים)
    const fetchSongs = async () => {
        try {
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
                            key={song.id} onClick={() => getSong(song.id, song.title)}
                        >
                            <div className="flex-fill">
                                <h5 style={{ marginBottom: '0px' }} className="ps-3"><strong>{song.title}</strong></h5>
                                {song.artist && (
                                    <p style={{ marginBottom: '0px' }} className="ps-3">
                                        <strong>Artist:</strong> {song.artist}</p>
                                )}

                                {/* Display tags only if there are tags */}
                                {song.tags && song.tags.length > 0 && (
                                    <p style={{ marginBottom: '0px' }} className="ps-3">
                                        <strong>Tags:</strong> {song.tags.join(', ')}
                                    </p>
                                )}
                            </div>
                            {/* Buttons container */}
                            <div className="d-flex justify-content-end">
                                {/* Delete icon button */}
                                <button
                                    onClick={() => handleDelete(song.id)}
                                    style={{
                                        padding: '5px',
                                        backgroundColor: 'transparent',
                                        border: 'none',
                                        cursor: 'pointer',
                                        fontSize: '20px',
                                    }}
                                    title="Delete">
                                    <MdDelete />
                                </button>
                                {/* Share icon button */}
                                <button
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
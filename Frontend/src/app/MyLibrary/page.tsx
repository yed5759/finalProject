"use client";

import React, { useState, useEffect } from 'react';
import { MdDelete, MdShare } from 'react-icons/md';

type Song = {
    id: string;
    title: string;
    artist?: string;
    notes: string[];
    tags?: string[];
};

export default function MyLibrary() {
    // songs list
    const [songs, setSongs] = useState<Song[]>([]); // Songs list state

    const [searchQuery, setSearchQuery] = useState(''); // State for search query

    // Filter songs based on search query
    const filteredSongs = songs.filter((song) =>
        song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (song.artist && song.artist.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (song.tags && song.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))) // if search query matches any tag
    );

    // Handle deleting a song widh given id
    const handleDelete = async (id: string) => {
        try {
            const res = await fetch(`http://localhost:5000/songs/${id}`, {
                method: "DELETE",
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
            });

            if (!res.ok) throw new Error("Failed to delete song");

            // Delete from client side
            setSongs(prevSongs => prevSongs.filter(song => song.id !== id));
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

    //todo delete this function after testing
    const handleAddConstSong = async () => {
        try {
            const res = await fetch("http://localhost:5000/songs/add-const", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`, // adjust this if your token is stored elsewhere
                },
            });

            if (!res.ok) throw new Error("Failed to add test song");

            const data = await res.json();
            setSongs(prevSongs => [...prevSongs, data.song]);
        } catch (error) {
            if (error instanceof Error) {
                alert("Error adding test song: " + error.message);
            } else {
                alert("An unknown error occurred.");
            }
        }

    };


    useEffect(() => {
        const fetchSongs = async () => {
            try {
                const res = await fetch("http://localhost:5000/songs", {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                    },
                });

                if (!res.ok) throw new Error("Failed to fetch songs");

                const data = await res.json();
                setSongs(data); // השרת מחזיר את רשימת השירים כ-array ישיר
            } catch (error) {
                console.error("Error fetching songs:", error);
                alert("שגיאה בטעינת רשימת השירים");
            }
        };

        fetchSongs();
    }, []);


    return (
        <>
            <h2 style={{ marginLeft: '70px' }}>My Library</h2>

            {/* Search bar */}
            <div style={{ margin: '20px 70px' }}>
                <input
                    type="text"
                    placeholder="Search for song, artist, or tag"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                        width: '100%',
                        padding: '10px',
                        fontSize: '16px',
                        borderRadius: '4px',
                        border: '1px solid #ccc',
                    }}
                />
            </div>

            {/* Add Const Song Button - מתחת לסרגל החיפוש */}
            <div style={{ margin: '0 70px 20px 70px' }}>
                <button
                    onClick={handleAddConstSong}
                    style={{
                        padding: '10px 20px',
                        fontSize: '16px',
                        borderRadius: '4px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                    }}
                >
                    Add Const Song
                </button>
            </div>

            <div style={{ maxHeight: '500px', overflowY: 'scroll', border: '1px solid #ccc', padding: '10px', margin: '30px 70px' }}>
                <ul style={{ listStyleType: 'none', padding: 0 }}>
                    {filteredSongs.map((song, index) => (
                        <li key={song.id} style={{ padding: '10px', borderBottom: index !== filteredSongs.length - 1 ? '1px solid black' : 'none' }}>
                            {/* Container for the song title, delete, and share buttons */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                {/* Song title */}
                                <h3 style={{ margin: 0 }}>{song.title}</h3>

                                {/* Buttons container */}
                                <div style={{ display: 'flex', gap: '10px' }}>
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
                                        title="Delete"
                                    >
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
                                        title="Share"
                                    >
                                        <MdShare />
                                    </button>
                                </div>
                            </div>

                            {/* Display artist only if available */}
                            {song.artist && (
                                <p style={{ marginBottom: '0px' }}><strong>Artist:</strong> {song.artist}</p>
                            )}

                            {/* Display tags only if there are tags */}
                            {song.tags && song.tags.length > 0 && (
                                <p style={{ marginBottom: '0px' }}>
                                    <strong>Tags:</strong> {song.tags.join(', ')}
                                </p>
                            )}
                        </li>
                    ))}
                </ul>
            </div>
        </>
    );
}

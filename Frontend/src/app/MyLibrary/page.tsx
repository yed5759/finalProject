"use client";

import React, { useState } from 'react';
import { MdDelete, MdShare } from 'react-icons/md';

// songs list
const initialSongs = [
    { id: '1', title: 'איש עם חץ', artist: 'Hillel', notes: ['Note 1', 'Note 2'], tags: ['pop', '2023'] },
    { id: '2', title: 'Song 2', artist: '', notes: ['Note 3', 'Note 4'], tags: ['rock', '2023'] },
    { id: '3', title: 'Sad Song', artist: 'Yedidya', notes: ['Note 5'], tags: [] },
    { id: '4', title: 'Summer Vibes', artist: 'Dana', notes: ['Note 6', 'Note 7'], tags: ['pop', '2022'] },
    { id: '5', title: 'Misty Night', artist: 'Michael', notes: ['Note 8'], tags: ['jazz', '2021'] },
    { id: '6', title: 'Deep Waters', artist: 'Sarah', notes: ['Note 9', 'Note 10'], tags: ['rock', '2022'] },
    { id: '7', title: 'City Lights', artist: 'Erez', notes: ['Note 11'], tags: ['electronic', '2023'] },
    { id: '8', title: 'Old Memories', artist: 'Yaara', notes: ['Note 12', 'Note 13'], tags: ['pop', '2021'] },
    { id: '9', title: 'The Sound of Silence', artist: 'Matan', notes: ['Note 14'], tags: ['indie', '2023'] },
    { id: '10', title: 'Shadows', artist: 'Noa', notes: ['Note 15', 'Note 16'], tags: ['rock', '2023'] },
];

export default function MyLibrary() {
    const [songs, setSongs] = useState(initialSongs); // Initial songs list state
    const [searchQuery, setSearchQuery] = useState(''); // State for search query

    // Filter songs based on search query
    const filteredSongs = songs.filter((song) =>
        song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (song.artist && song.artist.toLowerCase().includes(searchQuery.toLowerCase())) ||
        song.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) // if search query matches any tag
    );

    // Handle deleting a song
    const handleDelete = (id: string) => {
        // Remove song with the given id
        const updatedSongs = songs.filter(song => song.id !== id);
        setSongs(updatedSongs);
    };

    // Handle sharing a song (this is just a placeholder)
    const handleShare = (song: { title: string, artist: string }) => {
        // For now, just alert the song title and artist
        alert(`Sharing song: ${song.title} by ${song.artist}`);
    };

    //todo delete this place after testing
    const handleAddConstSong = async () => {
        try {
            const res = await fetch("http://localhost:5000/songs/add-const", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("token")}`, // adjust this if your token is stored elsewhere
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
                            {song.tags.length > 0 && (
                                <p style={{ marginBottom: '0px' }}><strong>Tags:</strong> {song.tags.join(', ')}</p>
                            )}
                        </li>
                    ))}
                </ul>
            </div>
        </>
    );
}

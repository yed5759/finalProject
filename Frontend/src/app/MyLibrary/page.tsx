"use client";

import React, { useState } from 'react';

// songs list
const songs = [
    {
        id: '1',
        title: 'איש עם חץ',
        artist: 'Hillel',
        notes: ['Note 1', 'Note 2'],
        tags: ['pop', '2023'],
    },
    {
        id: '2',
        title: 'Song 2',
        artist: '', // no artist
        notes: ['Note 3', 'Note 4'],
        tags: ['rock', '2023'],
    },
    {
        id: '3',
        title: 'Sad Song',
        artist: 'Yedidya',
        notes: ['Note 5'],
        tags: [], // no tags
    },
    {
        id: '4',
        title: 'Summer Vibes',
        artist: 'Dana',
        notes: ['Note 6', 'Note 7'],
        tags: ['pop', '2022'],
    },
    {
        id: '5',
        title: 'Misty Night',
        artist: 'Michael',
        notes: ['Note 8'],
        tags: ['jazz', '2021'],
    },
    {
        id: '6',
        title: 'Deep Waters',
        artist: 'Sarah',
        notes: ['Note 9', 'Note 10'],
        tags: ['rock', '2022'],
    },
    {
        id: '7',
        title: 'City Lights',
        artist: 'Erez',
        notes: ['Note 11'],
        tags: ['electronic', '2023'],
    },
    {
        id: '8',
        title: 'Old Memories',
        artist: 'Yaara',
        notes: ['Note 12', 'Note 13'],
        tags: ['pop', '2021'],
    },
    {
        id: '9',
        title: 'The Sound of Silence',
        artist: 'Matan',
        notes: ['Note 14'],
        tags: ['indie', '2023'],
    },
    {
        id: '10',
        title: 'Shadows',
        artist: 'Noa',
        notes: ['Note 15', 'Note 16'],
        tags: ['rock', '2023'],
    },
];

export default function MyLibrary() {
    const [searchQuery, setSearchQuery] = useState(''); // State for search query

    // filter songs based on search query
    const filteredSongs = songs.filter((song) =>
        song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (song.artist && song.artist.toLowerCase().includes(searchQuery.toLowerCase())) ||
        song.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) // if search query matches any tag
    );

    return (
        <>
            <h2 style={{ marginLeft: '60px' }}>My Library</h2>

            {/* search bar */}
            <div style={{ margin: '20px 60px' }}>
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

            <div style={{ maxHeight: '400px', overflowY: 'scroll', border: '1px solid #ccc', padding: '10px', margin: '60px' }}>
                <ul style={{ listStyleType: 'none', padding: 0 }}>
                    {filteredSongs.map((song, index) => (
                        <li key={song.id} style={{ padding: '10px', borderBottom: index !== filteredSongs.length - 1 ? '1px solid black' : 'none' }}>
                            <h3>{song.title}</h3>

                            {/* Display artist only if available */}
                            {song.artist && (
                                <p><strong>Artist:</strong> {song.artist}</p>
                            )}

                            {/* Display tags only if there are tags */}
                            {song.tags.length > 0 && (
                                <p><strong>Tags:</strong> {song.tags.join(', ')}</p>
                            )}
                        </li>
                    ))}
                </ul>
            </div>
        </>
    );
}

'use client';

import React, {useEffect, useRef} from 'react';

const bc = new BroadcastChannel("songs");

type Note = { keys: string[]; duration: string };

type CustomModalProps = {
    notes: Note[];
};

export default function CustomModal({notes}: CustomModalProps) {
    const formRef = useRef<HTMLFormElement>(null);
    const modalRef = useRef<HTMLElement | null>(null);
    const ModalClassRef = useRef<any>(null)

    useEffect(() => {
        // @ts-ignore
        import('bootstrap/dist/js/bootstrap.bundle.min.js').then((bootstrap) => {
            ModalClassRef.current = bootstrap.Modal;

            const modalEl = document.getElementById('staticBackdrop');
            if (modalEl) {
                modalRef.current = modalEl;

                modalEl.addEventListener('hidden.bs.modal', () => {
                    formRef.current?.reset();
                });
            }
        });
    }, []);

    const saveSong = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const form = formRef.current;
        if (!form) return;

        const formData = new FormData(form);
        const songName = formData.get("songName")?.toString().trim();
        const artist = formData.get("artist")?.toString().trim();

        if (!songName || !artist) {
            alert("Please enter both song name and artist name.");
            return;
        }

        try {
            const res = await fetch("http://localhost:5000/songs", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${localStorage.getItem("id_token")}`,
                },
                body: JSON.stringify({
                    title: songName,
                    artist: artist,
                    notes: notes,
                }),
            });

            if (!res.ok)
                throw new Error("Failed to add test song");

            bc.postMessage({ type: "song-added" });
        } catch (error) {
            alert("שגיאה בהוספת שיר: " + (error instanceof Error ? error.message : ""));
        }

        if (modalRef.current && ModalClassRef.current) {
            const modalInstance =
                ModalClassRef.current.getInstance(modalRef.current) ||
                new ModalClassRef.current(modalRef.current);

            modalInstance.hide();
        }
    };


    return (
        <div className="modal fade"
             style={{marginTop: "200px"}}
            id="staticBackdrop"
            data-bs-backdrop="static"
            data-bs-keyboard="false"
            // @ts-ignore
            tabIndex="-1"
            aria-labelledby="staticBackdropLabel"
            aria-hidden="true">
            <div className="modal-dialog">
                <form className="modal-content" onSubmit={saveSong} ref={formRef}>
                    <div className="modal-header">
                        <h1 className="modal-title fs-5" id="staticBackdropLabel">save as</h1>
                        <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div className="modal-body">
                        <label htmlFor="songName" className="form-label text-start w-100">Enter song name:</label>
                        <input name="songName" className="form-control form-control-sm" id="songName" type="text"
                               placeholder="there is no song in ba-sing-se"
                               aria-label=".form-control-sm example" required/>
                        <label htmlFor="artist" className="form-label text-start w-100 mt-1">Artist name:</label>
                        <input name="artist" className="form-control form-control-sm" id="artist" type="text"
                               placeholder="mr piano..."
                               aria-label=".form-control-sm example"/>
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-danger" data-bs-dismiss="modal">cancel</button>
                        <button type="submit" className="btn btn-secondary">save</button>
                    </div>
                </form>
            </div>
        </div>
    );
}
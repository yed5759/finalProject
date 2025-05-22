'use client';

import {useEffect, useRef} from 'react';

export default function CustomModal() {
    const inputRef = useRef<HTMLInputElement>(null);
    const modalRef = useRef<HTMLElement | null>(null);
    const ModalClassRef = useRef<any>(null)

    useEffect(() => {
        import('bootstrap/dist/js/bootstrap.bundle.min.js').then((bootstrap) => {
            ModalClassRef.current = bootstrap.Modal;

            const modalEl = document.getElementById('staticBackdrop');
            if (modalEl) {
                modalRef.current = modalEl;

                modalEl.addEventListener('show.bs.modal', () => {
                    if (inputRef.current) inputRef.current.value = '';
                });
            }
        });
    }, []);

    const saveSong = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const value = inputRef.current?.value?.trim();
        if (!value) {
            alert('Please enter a song name.');
            return;
        }

        console.log('Saving:', value);

        // Close modal programmatically
        if (modalRef.current && ModalClassRef.current) {
            const modalInstance =
                ModalClassRef.current.getInstance(modalRef.current) ||
                new ModalClassRef.current(modalRef.current);

            modalInstance.hide();
        }
    };


    return (
        <div
            className="modal fade"
            id="staticBackdrop"
            data-bs-backdrop="static"
            data-bs-keyboard="false"
            // @ts-ignore
            tabIndex="-1"
            aria-labelledby="staticBackdropLabel"
            aria-hidden="true"
        >
            <div className="modal-dialog">
                <form className="modal-content" onSubmit={saveSong}>
                    <div className="modal-header">
                        <h1 className="modal-title fs-5" id="staticBackdropLabel">save as</h1>
                        <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div className="modal-body">
                        <label htmlFor="songName" className="form-label text-start w-100">Enter song name:</label>
                        <input className="form-control form-control-sm" id="songName" type="text" placeholder="there is no song in ba-sing-se"
                               aria-label=".form-control-sm example" ref={inputRef} required/>
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
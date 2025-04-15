// src/app/landing/page.tsx
'use client';
import "../../styles/landing.css"
import Image from "next/image"
import { useRouter } from 'next/navigation'

export default function LandingPage() {
    console.log("🟡 [landing] landing page loaded"); // ✅ שורת הדפסה לזיהוי שהעמוד נטען

    const router = useRouter()


    const login = () => {
        const redirectUri = window.location.origin.replace('/landing', '/home');
        console.log('🟣 [landing] login redirectUri =', redirectUri);
        const loginUrl = `https://us-east-1ipj0spljz.auth.us-east-1.amazoncognito.com/login?client_id=1idr7soeln9lbch139sr9bmn8v&redirect_uri=${redirectUri}&response_type=code&scope=email+openid+phone`;
        router.push(loginUrl)
    }

    const register = () => {
        const redirectUri = window.location.origin.replace('/landing', '/home');
        const registerUrl = `https://us-east-1ipj0spljz.auth.us-east-1.amazoncognito.com/signup?client_id=1idr7soeln9lbch139sr9bmn8v&redirect_uri=${redirectUri}&response_type=code&scope=email+openid+phone`;
        router.push(registerUrl);
    }


    return (
        <div>
            <link href="https://fonts.googleapis.com/css2?family=Italianno&display=swap" rel="stylesheet" />
            <header id="stripe">
                <h4>Compose yourself and take notes!</h4>
                <p className="fs-6 fw-lighter"><small>Login to access the music notes generator</small></p>
            </header>
            <div className="container text-center overflow-hidden pt-lg-5">
                <div className="row d-flex align-items-stretch h-100">
                    <div className="col">
                        <Image src="/notes.png" alt="Notes" width={300} height={200} id="picture" />
                    </div>
                    <div className="col d-flex justify-content-center flex-column align-items-center text-center mb-1">
                        <h1><big>Generate Notes with Taking Notes</big></h1>
                        <h2>start now!</h2>
                        <button type="button" className="btn btn-dark button mt-3" onClick={login}>Log in</button>
                    </div>
                </div>
            </div>
            <div className="border-top border-dark" id="down">
                <div className="d-flex flex-column align-items-center justify-content-center mt-sm-5">
                    <h5>New User?</h5>
                    <p><small>Sign up and start using Taking Notes</small></p>
                    <button type="button" className="btn btn-dark button" onClick={register}>Sign Up</button>
                </div>
            </div>
        </div>
    );
}

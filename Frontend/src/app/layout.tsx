import "@/app/styles/globals.css";
import type {Metadata} from "next";

import 'bootstrap/dist/css/bootstrap.min.css';
import AuthWrapper from './AuthWrapper';
import AppWrapper from './AppWrapper';
import Image from "next/image";

export const metadata: Metadata = {
    title: "Create Next App",
    description: "Generated by create next app",
};
import {setupAmplify} from "@/app/utils/cognito";

setupAmplify();

export default function Layout({children}: { children: React.ReactNode }) {
    return (
        <html lang="en">
        <body>
        <div className="fixed top-0 left-0 h-screen w-[120px] z-[-1] pointer-events-none flex items-start">
            <Image
                src="/left.png"
                alt="left decoration"
                fill
                style={{ objectFit: 'contain', objectPosition: 'top left'}}
            />
        </div>
        <div className="fixed top-0 right-0 h-screen w-[120px] z-[-1] pointer-events-none flex items-start">
            <Image
                src="/right.png"
                alt="right decoration"
                fill
                style={{ objectFit: 'contain', objectPosition: 'top right' }}
            />
        </div>
        <AppWrapper>
            <AuthWrapper>
                {children}</AuthWrapper>
        </AppWrapper>
        </body>
        </html>
    );
}

// src/context/AuthContext.tsx

'use client';

// @ts-ignore
import { createContext, useContext, useState, ReactNode, useEffect, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import { isAuthenticated, exchangeCodeForToken } from '@/utils/cognito';

interface AuthContextType {
    isAuthenticated: boolean;
    loading: boolean;
    checkAuth: (code?: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [authState, setAuthState] = useState<boolean>(false);
    const [loading, setLoading] = useState<boolean>(true);
    const searchParams = useSearchParams();
    const usedCodeRef = useRef<string | null>(null);

    const checkAuth = async (code?: string) => {
        setLoading(true);

        if (usedCodeRef.current === code) {
            console.log('Code already used, skipping fetch');
            setLoading(false);
            return;
        }

        usedCodeRef.current = code || null;

        if (isAuthenticated()) {
            setAuthState(true);
        } else if (code) {
            const success = await exchangeCodeForToken(code);
            setAuthState(success);
        } else {
            setAuthState(isAuthenticated());
        }

        setLoading(false);
    };

    useEffect(() => {
        const code = searchParams.get('code');
        if (code) {
            checkAuth(code);
        } else {
            checkAuth();
        }
    }, [searchParams]);

    if (loading) {
        return null;
    }

    return (
        <AuthContext.Provider value={{ isAuthenticated: authState, loading, checkAuth }}>
            {children}
        </AuthContext.Provider>
    );
};

// src/utils/helpers.ts

/** Utility function to combine CSS classes */
export function cn(...inputs: (string | false | null | undefined)[]): string {
    return inputs.filter(Boolean).join(" ");
}
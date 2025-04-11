/**  Utility function to combine CSS classes */
export function cn(...inputs) {
  return inputs.filter(Boolean).join(" ");
}
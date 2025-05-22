/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export', // 🔥 this enables static export in App Router
    images: {
        unoptimized: true, // 👈 disables image optimization API
    }
}
module.exports = nextConfig
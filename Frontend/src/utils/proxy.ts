import { createProxyMiddleware } from 'http-proxy-middleware';

const proxy = createProxyMiddleware({
    target: 'http://localhost:5000', // הכתובת של Flask
    changeOrigin: true,
    pathRewrite: {
        '^/api': '', // מסיר את /api מהנתיב
    },
    cookieDomainRewrite: {
        '*': '', // מאפשר שליחת עוגיות לדומיין של Next.js
    }
});

export default proxy;

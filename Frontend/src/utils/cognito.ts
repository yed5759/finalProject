// // src/utils/cognito.ts
// import { Amplify } from 'aws-amplify';

// // Setup Amplify with environment variables
// export function setupAmplify() {
//   const {
//     USER_POOL_ID: userPoolId,
//     USER_POOL_CLIENT_ID: userPoolClientId,
//     IDENTITY_POOL_ID: identityPoolId
//   } = process.env

//   if (!(userPoolId && userPoolClientId && identityPoolId)) {
//     throw new Error('invalid ENV');
//   }

//   Amplify.configure({
//     Auth: {
//       Cognito: {
//         userPoolId,
//         userPoolClientId,
//         identityPoolId,
//         loginWith: {
//           email: true,
//         },
//         signUpVerificationMethod: "code",
//         userAttributes: {
//           email: {
//             required: true,
//           },
//         },
//         allowGuestAccess: true,
//         passwordFormat: {
//           minLength: 8,
//           requireLowercase: true,
//           requireUppercase: true,
//           requireNumbers: true,
//           requireSpecialCharacters: true,
//         }
//       },
//       region: 'us-east-1',
//       OAuth: {
//         // todo put domain
//         domain: 'your-domain.auth.us-east-1.amazoncognito.com',
//         redirectSignIn: 'http://localhost:3000/home',
//         redirectSignOut: 'http://localhost:3000/landing',
//         responseType: 'code',
//       },
//     } as any,
//   });
// }
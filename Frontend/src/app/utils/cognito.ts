// src/utils/cognito.ts

import {
  CognitoUserPool,
  CognitoUser,
  AuthenticationDetails,
} from 'amazon-cognito-identity-js';

import { Amplify } from 'aws-amplify';
import { generateClient } from 'aws-amplify/api';

// Optional: you can define the poolData once here
const poolData = {
  UserPoolId: 'us-east-1_IPJ0SpljZ',
  ClientId: '1idr7soeln9lbch139sr9bmn8v',
};

const userPool = new CognitoUserPool(poolData);

// ✅ Sign in with amazon-cognito-identity-js (if you need low-level control)
export const signIn = (username: string, password: string) => {
  const userData = { Username: username, Pool: userPool };
  const cognitoUser = new CognitoUser(userData);
  const authenticationDetails = new AuthenticationDetails({ Username: username, Password: password });

  return new Promise((resolve, reject) => {
    cognitoUser.authenticateUser(authenticationDetails, {
      onSuccess: (result) => {
        const idToken = result.getIdToken().getJwtToken();
        localStorage.setItem('accessToken', idToken);
        resolve(idToken);
      },
      onFailure: (err) => {
        console.error('Error during authentication', err);
        reject(err);
      },
    });
  });
};

// ✅ Modular Amplify configuration (v6)
export function setupAmplify() {
  const { USER_POOL_ID: userPoolId, USER_POOL_CLIENT_ID: userPoolClientId } = process.env;

  if (!(userPoolId && userPoolClientId)) throw new Error('invalid ENV');

  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId,
        userPoolClientId,
        identityPoolId: '<your-cognito-identity-pool-id>',
        loginWith: { email: true },
        signUpVerificationMethod: 'code',
        userAttributes: {
          email: { required: true },
        },
        allowGuestAccess: true,
        passwordFormat: {
          minLength: 8,
          requireLowercase: true,
          requireUppercase: true,
          requireNumbers: true,
          requireSpecialCharacters: true,
        },
      },
    },
  });
}

// ✅ Example API call using generateClient (modular)
export async function postToApi() {
  const token = localStorage.getItem('accessToken');
  const client = generateClient() as any;

  try {
    const result = await client.post({
      apiName: 'api', // this must match your Amplify REST API name
      path: '/x/y',
      options: {
        body: {
          key1: 'value1',
        },
        headers: {
          Authorization: `Bearer ${token}`,
        },
      },
    });
    return result;
  } catch (err) {
    console.error('API POST error', err);
    throw err;
  }
}
// src/utils/cognito.ts
import {
  CognitoUserPool,
  CognitoUser,
  AuthenticationDetails,
} from 'amazon-cognito-identity-js';
import { Amplify } from 'aws-amplify';
import API from 'aws-amplify/api';

// src/utils/cognito.ts

export function setupAmplify() {

  const {
    USER_POOL_ID: userPoolId,
    USER_POOL_CLIENT_ID: userPoolClientId,
    IDENTITY_POOL_ID: identityPoolId
  } = process.env

  if (!(userPoolId && userPoolClientId && identityPoolId)) throw new Error('invalid ENV');
  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId,
        userPoolClientId,
        identityPoolId,
        loginWith: {
          email: true,
        },
        signUpVerificationMethod: "code",
        userAttributes: {
          email: {
            required: true,
          },
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
  })

}

API.post({ apiName: 'api', path: '/x/y', options: {} })
const poolData = {
  UserPoolId: 'us-east-1_IPJ0SpljZ', // מזהה ה־User Pool שלך
  ClientId: '1idr7soeln9lbch139sr9bmn8v', // מזהה ה־App client שלך
};

const userPool = new CognitoUserPool(poolData);

export const signIn = (username: string, password: string) => {
  const userData = {
    Username: username,
    Pool: userPool,
  };

  const cognitoUser = new CognitoUser(userData);

  const authenticationData = {
    Username: username,
    Password: password,
  };

  const authenticationDetails = new AuthenticationDetails(authenticationData);

  return new Promise((resolve, reject) => {
    cognitoUser.authenticateUser(authenticationDetails, {
      onSuccess: (result) => {
        const idToken = result.getIdToken().getJwtToken(); // שליפת הטוקן
        localStorage.setItem('accessToken', idToken); // שמירת הטוקן ב־localStorage
        resolve(idToken);
      },
      onFailure: (err) => {
        console.error('Error during authentication', err);
        reject(err);
      },
    });
  });
};

// src/utils/cognito.js
import {
    CognitoUserPool,
    CognitoUser,
    AuthenticationDetails,
  } from 'amazon-cognito-identity-js';
  
  const poolData = {
    UserPoolId: 'your-user-pool-id', // מזהה ה־User Pool שלך
    ClientId: 'your-client-id', // מזהה ה־App client שלך
  };
  
  const userPool = new CognitoUserPool(poolData);
  
  export const signIn = (username, password) => {
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
  
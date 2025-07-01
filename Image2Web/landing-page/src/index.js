import { initializeApp } from "firebase/app";
import { getFirestore, doc, setDoc, serverTimestamp } from "firebase/firestore";
import {
  getAuth,
  onAuthStateChanged,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut
} from "firebase/auth";
import { getAnalytics, logEvent } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyDA9TT-iCH515H63y-4zJkTJ1tZAOU7m60",
  authDomain: "image2web-1a6c0.firebaseapp.com",
  projectId: "image2web-1a6c0",
  storageBucket: "image2web-1a6c0.firebasestorage.app",
  messagingSenderId: "167811907825",
  appId: "1:167811907825:web:809c410fefc52de3542950",
  measurementId: "G-LF3Y1S5M47"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);
const analytics = getAnalytics(app);

// UI elements
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const signupBtn = document.getElementById("signup-btn");
const loginBtn = document.getElementById("login-btn");
const logoutBtn = document.getElementById("logout-btn");
const messageDiv = document.getElementById("message");
const profileDiv = document.getElementById("profile");

// Sign up
signupBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  const email = emailInput.value;
  const password = passwordInput.value;
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    await setDoc(doc(db, "users", userCredential.user.uid), {
      email: userCredential.user.email,
      createdAt: serverTimestamp()
    });
    logEvent(analytics, 'sign_up', { method: 'password' });
    messageDiv.textContent = "Account created!";
  } catch (err) {
    messageDiv.textContent = err.message;
  }
});

// Log in
loginBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  const email = emailInput.value;
  const password = passwordInput.value;
  try {
    await signInWithEmailAndPassword(auth, email, password);
    logEvent(analytics, 'login', { method: 'password' });
    messageDiv.textContent = "Logged in!";
  } catch (err) {
    messageDiv.textContent = err.message;
  }
});

// Log out
logoutBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  await signOut(auth);
  messageDiv.textContent = "Logged out!";
});

// Show user profile and toggle logout button
onAuthStateChanged(auth, (user) => {
  if (user) {
    profileDiv.textContent = `Logged in as: ${user.email}`;
    logoutBtn.style.display = "inline";
  } else {
    profileDiv.textContent = "";
    logoutBtn.style.display = "none";
  }
});
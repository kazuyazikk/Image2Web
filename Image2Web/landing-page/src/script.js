// script.js

// Firebase Auth reference
const auth = firebase.auth();

// DOM elements
const signupModal = document.getElementById('signupModal');
const loginModal = document.getElementById('loginModal');
const aboutModal = document.getElementById('aboutModal');

const userGreeting = document.getElementById('user-greeting');
const userNameSpan = document.getElementById('user-name');
const authLinks = document.getElementById('auth-links');
const userLinks = document.getElementById('user-links');

const signupForm = document.getElementById('signupForm');
const loginForm = document.getElementById('loginForm');
const successMessage = document.getElementById('successMessage');

// Open modals
function openSignup() {
  signupModal.style.display = 'flex';
}
function closeSignup() {
  signupModal.style.display = 'none';
}
function openLogin() {
  loginModal.style.display = 'flex';
}
function closeLogin() {
  loginModal.style.display = 'none';
}
function openAbout() {
  aboutModal.style.display = 'flex';
}
function closeAbout() {
  aboutModal.style.display = 'none';
}

// Sign up form submit
signupForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const username = document.getElementById('signup-username').value;
  const email = document.getElementById('signup-email').value;
  const password = document.getElementById('signup-password').value;

  auth.createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      return user.updateProfile({ displayName: username });
    })
    .then(() => {
      successMessage.style.display = 'block';
      signupForm.reset();
      setTimeout(() => {
        successMessage.style.display = 'none';
        closeSignup();
      }, 2000);
    })
    .catch((error) => {
      alert(error.message);
    });
});

// Login form submit
loginForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const email = document.getElementById('login-username').value;
  const password = document.getElementById('login-password').value;

  auth.signInWithEmailAndPassword(email, password)
    .then(() => {
      loginForm.reset();
      closeLogin();
    })
    .catch((error) => {
      alert(error.message);
    });
});

// Logout button
function logout() {
  auth.signOut().catch((error) => {
    alert(error.message);
  });
}

// Auth state listener
auth.onAuthStateChanged((user) => {
  if (user) {
    userGreeting.style.display = 'block';
    userNameSpan.textContent = user.displayName || user.email;
    authLinks.style.display = 'none';
    userLinks.style.display = 'block';
  } else {
    userGreeting.style.display = 'none';
    userNameSpan.textContent = '';
    authLinks.style.display = 'block';
    userLinks.style.display = 'none';
  }
});

// Close modal if clicked outside
window.addEventListener('click', (e) => {
  if (e.target === signupModal) closeSignup();
  if (e.target === loginModal) closeLogin();
  if (e.target === aboutModal) closeAbout();
});

// This file contains the JavaScript code for the landing page.
// It handles user interactions, dynamic content updates, and other client-side logic.

// --- Modal UI logic ---
function openSignup() {
    document.getElementById('signupModal').style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeSignup() {
    document.getElementById('signupModal').style.display = 'none';
    document.getElementById("successMessage").style.display = "none"; // Ensure success message is hidden
    document.body.style.overflow = 'auto';
}

function openLogin() {
    document.getElementById("loginModal").style.display = "block";
    document.body.style.overflow = "hidden";
}
function closeLogin() {
    document.getElementById("loginModal").style.display = "none";
    document.body.style.overflow = "auto";
}
function openAbout() {
    document.getElementById("aboutModal").style.display = "block";
}
function closeAbout() {
    document.getElementById("aboutModal").style.display = "none";
}

// Forgot Password modal logic
function openForgotPassword() {
    document.getElementById('forgotPasswordModal').style.display = 'block';
    document.body.style.overflow = 'hidden';
}
function closeForgotPassword() {
    document.getElementById('forgotPasswordModal').style.display = 'none';
    document.body.style.overflow = 'auto';
    // Hide messages
    const succ = document.getElementById('forgot-success');
    const err = document.getElementById('forgot-error');
    if (succ) succ.style.display = 'none';
    if (err) err.style.display = 'none';
}

// Optional: Close modals when clicking outside of them
window.onclick = function(event) {
    if (event.target.classList && event.target.classList.contains("signup-modal")) closeSignup();
    if (event.target.classList && event.target.classList.contains("login-modal")) closeLogin();
    if (event.target.classList && event.target.classList.contains("about-modal")) closeAbout();
};

// Navigation logic
function navigate(page) {
    const content = document.getElementById('content');
    let html = '';
    switch (page) {
        case 'home':
            html = `
                <section class="hero">
                  <div class="hero-content">
                    <div class="hero-label">Digital Insights</div>
                    <h1 class="hero-title">The Future<br>of Technology</h1>
                    <button class="hero-btn" onclick="navigate('get-started')">Get started</button>
                  </div>
                </section>
            `;
            break;
        case 'contributors':
            html = `
                <section class="hero">
                  <div class="hero-content">
                    <div class="hero-label">Contributors</div>
                    <ul class="contributors-list">
                      <li>Moyano, Sarge Dave M.</li>
                      <li>Mallari, Merick Joshua</li>
                      <li>Quiambao, Christian Joshua</li>
                      <li>Dizon, Robby</li>
                    </ul>
                  </div>
                </section>
            `;
            break;
        case 'get-started':
            html = `
                <section class="hero">
                  <div class="hero-content">
                    <div class="hero-label">Get Started</div>
                    <p>Welcome! Explore our resources and join our community.</p>
                  </div>
                </section>
            `;
            break;
        default:
            html = `
                <section>
                  <h2>Page Not Found</h2>
                  <p>The page you are looking for does not exist.</p>
                </section>
            `;
    }
    if (content) content.innerHTML = html;

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    // Make sure your nav-items in HTML have data-page attributes, e.g., <li class="nav-item" data-page="home">
    const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (nav) nav.classList.add('active');

    // Close the mobile menu if it's open after navigating
    const navMenu = document.getElementById('nav-menu');
    if (navMenu && navMenu.classList.contains('open')) {
        navMenu.classList.remove('open');
        const navToggle = document.getElementById('nav-toggle');
        if (navToggle) navToggle.setAttribute('aria-expanded', false);
    }
}
window.navigate = navigate; // Make it available globally


// --- LOGOUT ---
function logout() {
    firebase.auth().signOut().then(() => {
        alert("Logged out!");
        // Update UI for logged out state
        const authLinks = document.getElementById("auth-links");
        const userLinks = document.getElementById("user-links");
        const greetingDiv = document.getElementById("user-greeting");

        if (authLinks) authLinks.style.display = "block";
        if (userLinks) userLinks.style.display = "none";
        if (greetingDiv) greetingDiv.style.display = "none";
    }).catch((error) => {
        console.error("Error logging out:", error);
        alert("Error logging out: " + error.message);
    });
}
window.logout = logout; // Make it globally accessible if called from HTML

// --- Show/hide nav links and user greeting based on auth state ---
// Ensure Firebase SDKs are loaded before this runs
if (typeof firebase !== 'undefined' && firebase.auth) {
    firebase.auth().onAuthStateChanged(function(user) {
        const greetingDiv = document.getElementById("user-greeting");
        const nameSpan = document.getElementById("user-name");
        const authLinks = document.getElementById("auth-links");
        const userLinks = document.getElementById("user-links");

        if (user) {
            if (authLinks) authLinks.style.display = "none";
            if (userLinks) userLinks.style.display = "list-item"; // Assuming this is an li

            if (greetingDiv && nameSpan) {
                // Always fetch username from Firestore and display it
                firebase.firestore().collection("users").doc(user.uid).get().then(doc => {
                    if (doc.exists && doc.data().username) {
                        nameSpan.textContent = doc.data().username;
                    } else {
                        nameSpan.textContent = "User"; // fallback if username missing
                    }
                    greetingDiv.style.display = "block";
                }).catch((error) => {
                    console.error("Error fetching username:", error);
                    nameSpan.textContent = "User"; // Fallback on error
                    greetingDiv.style.display = "block";
                });
            }
        } else {
            if (authLinks) authLinks.style.display = "list-item"; // Assuming this is an li
            if (userLinks) userLinks.style.display = "none";
            if (greetingDiv) greetingDiv.style.display = "none";
        }
    });
}


// --- DOMContentLoaded for main event listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Forgot Password form handler
    const forgotForm = document.getElementById('forgotPasswordForm');
    if (forgotForm) {
        forgotForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const email = document.getElementById('forgot-email').value;
            const succ = document.getElementById('forgot-success');
            const err = document.getElementById('forgot-error');
            if (succ) succ.style.display = 'none';
            if (err) err.style.display = 'none';
            firebase.auth().sendPasswordResetEmail(email)
                .then(() => {
                    if (succ) succ.style.display = 'block';
                })
                .catch((error) => {
                    if (err) {
                        err.textContent = error.message;
                        err.style.display = 'block';
                    }
                });
        });
    }
    // Initial load
    // Check if 'content' element exists before trying to navigate, as it might not be present on all pages (e.g., workspace.html)
    if (document.getElementById('content')) {
        navigate('home');
    }

    // --- Hamburger Menu Toggle ---
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('open');
            // Toggle aria-expanded for accessibility
            const isExpanded = navMenu.classList.contains('open');
            navToggle.setAttribute('aria-expanded', isExpanded);
        });

        // Optional: Close menu when a nav item is clicked
        navMenu.querySelectorAll('.nav-item button').forEach(button => {
            button.addEventListener('click', () => {
                navMenu.classList.remove('open');
                navToggle.setAttribute('aria-expanded', false);
            });
        });
    }

    // --- Google Sign-In ---
    const googleBtn = document.getElementById('google-signin-btn');
    if (googleBtn) {
        googleBtn.addEventListener('click', function() {
            const provider = new firebase.auth.GoogleAuthProvider();
            firebase.auth().signInWithPopup(provider)
                .then((result) => {
                    // Save user info to Firestore if new
                    const user = result.user;
                    if (user) {
                        const usersRef = firebase.firestore().collection('users').doc(user.uid);
                        usersRef.get().then(doc => {
                            if (!doc.exists) {
                                usersRef.set({
                                    username: user.displayName || 'Google User',
                                    email: user.email,
                                    createdAt: new Date(),
                                    provider: 'google'
                                });
                            }
                        });
                    }
                    closeLogin();
                    if (typeof firebase.analytics !== 'undefined') {
                        firebase.analytics().logEvent('login', { method: 'google' });
                    }
                    alert('Logged in with Google!');
                })
                .catch((error) => {
                    alert('Google Sign-In Error: ' + error.message);
                    console.error('Google Sign-In Error:', error);
                });
        });
    }


    // Signup form handler
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const username = document.getElementById('signup-username').value;
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;

            firebase.auth().createUserWithEmailAndPassword(email, password)
                .then((userCredential) => {
                    const user = userCredential.user;
                    // Save username to Firestore
                    return firebase.firestore().collection('users').doc(user.uid).set({
                        username: username,
                        email: email,
                        createdAt: new Date()
                    });
                })
                .then(() => {
                    document.getElementById('successMessage').style.display = 'block';
                    setTimeout(() => {
                        closeSignup();
                    }, 2000);
                    // Log signup event to Firebase Analytics if available
                    if (typeof firebase.analytics !== 'undefined') {
                        firebase.analytics().logEvent('sign_up', { method: 'email_password' });
                    }
                })
                .catch((error) => {
                    alert("Signup Error: " + error.message);
                    console.error("Firebase Auth Signup Error:", error);
                });
        });
    }

    // Login form handler
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const loginEmail = document.getElementById('login-username').value; // Assuming 'login-username' is actually email
            const loginPassword = document.getElementById('login-password').value;

            firebase.auth().signInWithEmailAndPassword(loginEmail, loginPassword)
                .then((userCredential) => {
                    closeLogin();
                    // Log login event to Firebase Analytics if available
                    if (typeof firebase.analytics !== 'undefined') {
                        firebase.analytics().logEvent('login', { method: 'email_password' });
                    }
                    alert("Logged in successfully!");
                })
                .catch((error) => {
                    alert("Login Error: " + error.message);
                    console.error("Firebase Auth Login Error:", error);
                });
        });
    }
});

// --- Spinner functions (make sure you have an element with id 'loading-spinner' in your HTML) ---
function showSpinner() {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) spinner.style.display = 'flex';
}
function hideSpinner() {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) spinner.style.display = 'none';
}
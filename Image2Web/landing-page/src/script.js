// This file contains the JavaScript code for the landing page. 
// It handles user interactions, dynamic content updates, and other client-side logic.

function openSignup(){
  document.getElementById('signupModal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}

function closeSignup(){
  document.getElementById('signupModal').style.display = 'none';
  document.body.style.overflow = 'auto';
}

// Close the modal when clicking outside
window.onclick = function(event) {
  const modal = document.getElementById('signupModal');
  if(event.target === modal){
    closeSignup();
  }
}

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
                      <li>DIzon, Robby</li>
                    </ul>
                  </div>
                </section>
            `;
            break;
        case 'subscribe':
            html = `
                <section class="hero">
                  <div class="hero-content">
                    <div class="hero-label">Subscribe</div>
                    <form class="subscribe-form" onsubmit="event.preventDefault();alert('Thank you for subscribing!');">
                      <input type="email" placeholder="Your email" required />
                      <button type="submit">Subscribe</button>
                    </form>
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
    content.innerHTML = html;

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (nav) nav.classList.add('active');
}

// Function to handle user logout
function logout() {
  firebase.auth().signOut().then(() => {
    console.log("User logged out");
    alert("Logged out successfully!");
  }).catch((error) => {
    console.error("Error logging out:", error);
  });
}

window.navigate = navigate; // Make it available globally

document.addEventListener('DOMContentLoaded', () => {
    // Initial load
    navigate('home');

    // Signup form handler
    
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
      signupForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const username = document.getElementById('signup-username').value;
        const email = document.getElementById('signup-email').value;
        const password = document.getElementById('signup-password').value;

        // Create user with Firebase Authentication
        firebase.auth().createUserWithEmailAndPassword(email, password)
          .then((userCredential) => {
            // Signed in successfully
            const user = userCredential.user;

            // Save username to Firestore (assuming you have Firestore initialized)
            firebase.firestore().collection('users').doc(user.uid).set({
              username: username,
              email: user.email
            })
            .then(() => {
              console.log("User profile saved to Firestore!");
              // Display success message and close modal
              document.getElementById('successMessage').style.display = 'block';
              // You might want to automatically log in the user after signup
              // Log signup event to Firebase Analytics
 firebase.analytics().logEvent('sign_up', { method: 'email_password' });
              // and redirect them to a different page or update the UI
              // For now, let's just close the modal after a delay
              setTimeout(() => {
                closeSignup();
              }, 3000); // Close after 3 seconds
            })
            .catch((error) => {
              console.error("Error saving user profile:", error);
              // Display an error message to the user
              alert("Error saving user profile: " + error.message);
            });
          })
          .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error("Firebase Authentication Error:", errorCode, errorMessage);
            // Display an error message to the user
            alert("Signup Error: " + errorMessage);
          });
      });
    }

    // Login form handler
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
      loginForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const loginUsernameOrEmail = document.getElementById('login-username').value;
        const loginPassword = document.getElementById('login-password').value;

        // Sign in user with Firebase Authentication
        firebase.auth().signInWithEmailAndPassword(loginUsernameOrEmail, loginPassword)
          .then((userCredential) => {
            // Signed in successfully
            const user = userCredential.user;
            console.log("User logged in:", user);
            // Close login modal
            closeLogin();
            // You can update the UI here to show logged-in state
            // and potentially redirect the user
            // Log login event to Firebase Analytics
 firebase.analytics().logEvent('login', { method: 'email_password' });
            alert("Logged in successfully!");
          })
          .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error("Firebase Authentication Error:", errorCode, errorMessage);
            // Display an error message to the user
            alert("Login Error: " + errorMessage);
          });
      });
    }
});

function openAbout() {
  document.getElementById('aboutModal').classList.add('active');
  document.body.style.overflow = 'hidden';
}
function closeAbout() {
  document.getElementById('aboutModal').classList.remove('active');
  document.body.style.overflow = '';
}
function openLogin() {
  document.getElementById('loginModal').classList.add('active');
  document.body.style.overflow = 'hidden';
}
function closeLogin() {
 document.getElementById('loginModal').classList.remove('active');
 document.body.style.overflow = '';
}

// --- Modal UI logic (keep your existing open/close functions) ---

// --- SIGN UP ---
document.getElementById("signupForm").addEventListener("submit", function(e) {
  e.preventDefault();
  const email = document.getElementById("signup-email").value;
  const password = document.getElementById("signup-password").value;
  firebase.auth().createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      document.getElementById("successMessage").style.display = "block";
      // Optionally, store user profile in Firestore
      firebase.firestore().collection("users").doc(userCredential.user.uid).set({
        email: email,
        createdAt: new Date()
      });
    })
    .catch((error) => {
      alert(error.message);
    });
});

// --- LOGIN ---
document.getElementById("loginForm").addEventListener("submit", function(e) {
  e.preventDefault();
  const emailOrUsername = document.getElementById("login-username").value;
  const password = document.getElementById("login-password").value;
  // For demo, treat input as email
  firebase.auth().signInWithEmailAndPassword(emailOrUsername, password)
    .then(() => {
      alert("Logged in!");
      closeLogin();
      document.getElementById("auth-links").style.display = "none";
      document.getElementById("user-links").style.display = "block";
    })
    .catch((error) => {
      alert(error.message);
    });
});

// --- LOGOUT ---
function logout() {
  firebase.auth().signOut().then(() => {
    alert("Logged out!");
    document.getElementById("auth-links").style.display = "block";
    document.getElementById("user-links").style.display = "none";
  });
}

// --- Show/hide nav links based on auth state ---
firebase.auth().onAuthStateChanged(function(user) {
  if (user) {
    document.getElementById("auth-links").style.display = "none";
    document.getElementById("user-links").style.display = "block";
  } else {
    document.getElementById("auth-links").style.display = "block";
    document.getElementById("user-links").style.display = "none";
  }
});
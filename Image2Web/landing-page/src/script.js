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

// --- Modal UI logic ---
function openSignup() {
  document.getElementById("signupModal").style.display = "block";
  document.body.style.overflow = "hidden";
}
function closeSignup() {
  document.getElementById("signupModal").style.display = "none";
  document.getElementById("successMessage").style.display = "none";
  document.body.style.overflow = "auto";
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
  if (content) content.innerHTML = html;

  // Update active nav item
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (nav) nav.classList.add('active');
}
window.navigate = navigate;

// --- SIGN UP ---
document.addEventListener('DOMContentLoaded', () => {
  // Initial load
  if (document.getElementById('content')) navigate('home');

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
          firebase.firestore().collection('users').doc(user.uid).set({
            username: username,
            email: user.email,
            createdAt: new Date()
          })
          .then(() => {
            document.getElementById('successMessage').style.display = 'block';
            firebase.analytics().logEvent('sign_up', { method: 'email_password' });
            setTimeout(() => {
              closeSignup();
            }, 2000);
          })
          .catch((error) => {
            alert("Error saving user profile: " + error.message);
          });
        })
        .catch((error) => {
          alert("Signup Error: " + error.message);
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

      firebase.auth().signInWithEmailAndPassword(loginUsernameOrEmail, loginPassword)
        .then((userCredential) => {
          closeLogin();
          firebase.analytics().logEvent('login', { method: 'email_password' });
          alert("Logged in successfully!");
        })
        .catch((error) => {
          alert("Login Error: " + error.message);
        });
    });
  }
});

// --- LOGOUT ---
function logout() {
  firebase.auth().signOut().then(() => {
    alert("Logged out!");
    document.getElementById("auth-links").style.display = "block";
    document.getElementById("user-links").style.display = "none";
    // Hide greeting
    const greetingDiv = document.getElementById("user-greeting");
    if (greetingDiv) greetingDiv.style.display = "none";
  });
}

// --- Show/hide nav links and user greeting based on auth state ---
firebase.auth().onAuthStateChanged(function(user) {
  const greetingDiv = document.getElementById("user-greeting");
  const nameSpan = document.getElementById("user-name");
  console.log("Auth state changed. User:", user);
  if (user) {
    document.getElementById("auth-links").style.display = "none";
    document.getElementById("user-links").style.display = "block";
    firebase.firestore().collection("users").doc(user.uid).get().then(doc => {
      let username = doc.exists && doc.data().username ? doc.data().username : user.email;
      console.log("Fetched username:", username);
      if (nameSpan) nameSpan.textContent = username;
      if (greetingDiv) greetingDiv.style.display = "block";
    });
  } else {
    document.getElementById("auth-links").style.display = "block";
    document.getElementById("user-links").style.display = "none";
    if (greetingDiv) greetingDiv.style.display = "none";
  }
});

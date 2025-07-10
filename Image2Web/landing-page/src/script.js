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

// --- Optional: Close modals when clicking outside ---
window.onclick = function (event) {
  if (event.target.classList.contains("signup-modal")) closeSignup();
  if (event.target.classList.contains("login-modal")) closeLogin();
  if (event.target.classList.contains("about-modal")) closeAbout();
};

// --- Navigation logic ---
function navigate(page) {
  const content = document.getElementById('content');
  if (!content) return; // Skip if no content container

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
  content.innerHTML = html;

  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (nav) nav.classList.add('active');
}
window.navigate = navigate;

// --- Auth logic (signup, login, logout) ---
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('content')) navigate('home');

  // Signup form
  const signupForm = document.getElementById('signupForm');
  if (signupForm) {
    signupForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const username = document.getElementById('signup-username').value;
      const email = document.getElementById('signup-email').value;
      const password = document.getElementById('signup-password').value;

      firebase.auth().createUserWithEmailAndPassword(email, password)
        .then((userCredential) => {
          const user = userCredential.user;
          return firebase.firestore().collection('users').doc(user.uid).set({
            username: username,
            email: email,
            createdAt: new Date()
          });
        })
        .then(() => {
          document.getElementById('successMessage').style.display = 'block';
          firebase.analytics().logEvent('sign_up', { method: 'email_password' });
          setTimeout(() => {
            closeSignup();
          }, 2000);
        })
        .catch((error) => {
          alert("Signup Error: " + error.message);
        });
    });
  }

  // Login form
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const email = document.getElementById('login-username').value;
      const password = document.getElementById('login-password').value;

      firebase.auth().signInWithEmailAndPassword(email, password)
        .then(() => {
          firebase.analytics().logEvent('login', { method: 'email_password' });
          closeLogin();
          alert("Logged in successfully!");
        })
        .catch((error) => {
          alert("Login Error: " + error.message);
        });
    });
  }
});

// --- Logout function ---
function logout() {
  firebase.auth().signOut().then(() => {
    alert("Logged out!");
    document.getElementById("auth-links").style.display = "block";
    document.getElementById("user-links").style.display = "none";
    document.getElementById("user-greeting").style.display = "none";
  }).catch((error) => {
    console.error("Error logging out:", error);
  });
}

// --- Auth state observer ---
firebase.auth().onAuthStateChanged(function (user) {
  const greetingDiv = document.getElementById("user-greeting");
  const nameSpan = document.getElementById("user-name");
  if (user) {
    document.getElementById("auth-links").style.display = "none";
    document.getElementById("user-links").style.display = "list-item";

    firebase.firestore().collection("users").doc(user.uid).get().then(doc => {
      if (doc.exists && doc.data().username) {
        nameSpan.textContent = doc.data().username;
      } else {
        nameSpan.textContent = "User";
      }
      greetingDiv.style.display = "block";
    }).catch(() => {
      nameSpan.textContent = "User";
      greetingDiv.style.display = "block";
    });
  } else {
    document.getElementById("auth-links").style.display = "list-item";
    document.getElementById("user-links").style.display = "none";
    greetingDiv.style.display = "none";
  }
});

// --- Navbar toggle (if using) ---
const navToggle = document.getElementById('nav-toggle');
if (navToggle) {
  navToggle.onclick = function () {
    document.querySelector('nav ul').classList.toggle('open');
  };
}

// --- Spinner utility (optional) ---
function showSpinner() {
  document.getElementById('loading-spinner').style.display = 'flex';
}
function hideSpinner() {
  document.getElementById('loading-spinner').style.display = 'none';
}

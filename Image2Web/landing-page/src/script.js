// =================== MODALS =================== //

// Signup
function openSignup() {
  const el = document.getElementById('signupModal');
  if (el) { el.style.display = 'flex'; document.body.style.overflow = 'hidden'; }
}
function closeSignup() {
  const el = document.getElementById('signupModal');
  if (el) el.style.display = 'none';
  const ok = document.getElementById('successMessage');
  if (ok) ok.style.display = 'none';
  document.body.style.overflow = 'auto';
}

// Login
function openLogin() {
  const el = document.getElementById('loginModal');
  if (el) { el.style.display = 'flex'; document.body.style.overflow = 'hidden'; }
}
function closeLogin() {
  const el = document.getElementById('loginModal');
  if (el) el.style.display = 'none';
  document.body.style.overflow = 'auto';
}

// About
function openAbout() {
  const el = document.getElementById('aboutModal');
  if (el) el.style.display = 'flex';
}
function closeAbout() {
  const el = document.getElementById('aboutModal');
  if (el) el.style.display = 'none';
}

// Forgot Password
function openForgotPassword() {
  // Close login first (so we don't show two modals)
  closeLogin();
  const el = document.getElementById('forgotPasswordModal');
  if (el) {
    el.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    const input = document.getElementById('forgot-email');
    if (input) input.focus();
  } else {
    alert('Forgot Password modal not found. Please add it to your HTML.');
  }
}
function closeForgotPassword() {
  const el = document.getElementById('forgotPasswordModal');
  if (el) el.style.display = 'none';
  document.body.style.overflow = 'auto';
  const succ = document.getElementById('forgot-success');
  const err = document.getElementById('forgot-error');
  if (succ) succ.style.display = 'none';
  if (err) err.style.display = 'none';
}

// Make callable from inline HTML onclick=""
window.openSignup = openSignup;
window.closeSignup = closeSignup;
window.openLogin = openLogin;
window.closeLogin = closeLogin;
window.openAbout = openAbout;
window.closeAbout = closeAbout;
window.openForgotPassword = openForgotPassword;
window.closeForgotPassword = closeForgotPassword;

// Close modals by clicking on the overlay only (not inside content)
window.addEventListener('click', (e) => {
  if (e.target?.id === 'signupModal') closeSignup();
  if (e.target?.id === 'loginModal') closeLogin();
  if (e.target?.id === 'forgotPasswordModal') closeForgotPassword();
  if (e.target?.id === 'aboutModal') closeAbout();
});


// =================== NAVIGATION (optional SPA slots) =================== //
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
        </section>`;
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
        </section>`;
      break;
    case 'get-started':
      html = `
        <section class="hero">
          <div class="hero-content">
            <div class="hero-label">Get Started</div>
            <p>Welcome! Explore our resources and join our community.</p>
          </div>
        </section>`;
      break;
    default:
      html = `
        <section>
          <h2>Page Not Found</h2>
          <p>The page you are looking for does not exist.</p>
        </section>`;
  }

  if (content) content.innerHTML = html;

  // Update active nav item
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  const nav = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (nav) nav.classList.add('active');

  // Close mobile menu if open
  const navMenu = document.getElementById('nav-menu');
  const navToggle = document.getElementById('nav-toggle');
  if (navMenu?.classList.contains('open')) {
    navMenu.classList.remove('open');
    if (navToggle) navToggle.setAttribute('aria-expanded', 'false');
  }
}
window.navigate = navigate;


// =================== AUTH =================== //
function logout() {
  firebase.auth().signOut().then(() => {
    alert("Logged out!");
    const authLinks = document.getElementById("auth-links");
    const userLinks = document.getElementById("user-links");
    const greetingDiv = document.getElementById("user-greeting");
    if (authLinks) authLinks.style.display = "list-item";
    if (userLinks) userLinks.style.display = "none";
    if (greetingDiv) greetingDiv.style.display = "none";
  }).catch((error) => {
    console.error("Error logging out:", error);
    alert("Error logging out: " + error.message);
  });
}
window.logout = logout;

// Reflect auth state in header
if (typeof firebase !== 'undefined' && firebase.auth) {
  firebase.auth().onAuthStateChanged(function(user) {
    const greetingDiv = document.getElementById("user-greeting");
    const nameSpan = document.getElementById("user-name");
    const authLinks = document.getElementById("auth-links");
    const userLinks = document.getElementById("user-links");

    if (user) {
      if (authLinks) authLinks.style.display = "none";
      if (userLinks) userLinks.style.display = "list-item";
      if (greetingDiv && nameSpan) {
        firebase.firestore().collection("users").doc(user.uid).get().then(doc => {
          nameSpan.textContent = (doc.exists && doc.data().username) ? doc.data().username : "User";
          greetingDiv.style.display = "block";
        }).catch(() => {
          nameSpan.textContent = "User";
          greetingDiv.style.display = "block";
        });
      }
    } else {
      if (authLinks) authLinks.style.display = "list-item";
      if (userLinks) userLinks.style.display = "none";
      if (greetingDiv) greetingDiv.style.display = "none";
    }
  });
}


// =================== DOMContentLoaded =================== //
document.addEventListener('DOMContentLoaded', () => {
  // Initial SPA section (only if #content exists)
  if (document.getElementById('content')) navigate('home');

  // Hamburger
  const navToggle = document.getElementById('nav-toggle');
  const navMenu = document.getElementById('nav-menu');
  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', navMenu.classList.contains('open') ? 'true' : 'false');
    });
    navMenu.querySelectorAll('.nav-item button').forEach(btn => {
      btn.addEventListener('click', () => {
        navMenu.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Google Sign-In
  const googleBtn = document.getElementById('google-signin-btn');
  if (googleBtn) {
    googleBtn.addEventListener('click', function() {
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider).then((result) => {
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
      }).catch((error) => {
        alert('Google Sign-In Error: ' + error.message);
        console.error('Google Sign-In Error:', error);
      });
    });
  }

  // Signup form
  const signupForm = document.getElementById('signupForm');
  if (signupForm) {
    signupForm.addEventListener('submit', function(e) {
      e.preventDefault();
      const username = document.getElementById('signup-username').value;
      const email = document.getElementById('signup-email').value;
      const password = document.getElementById('signup-password').value;

      firebase.auth().createUserWithEmailAndPassword(email, password)
        .then((cred) => {
          const user = cred.user;
          return firebase.firestore().collection('users').doc(user.uid).set({
            username, email, createdAt: new Date()
          });
        })
        .then(() => {
          const ok = document.getElementById('successMessage');
          if (ok) ok.style.display = 'block';
          setTimeout(() => closeSignup(), 2000);
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

  // Login form
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', function(e) {
      e.preventDefault();
      const email = document.getElementById('login-username').value;
      const password = document.getElementById('login-password').value;
      firebase.auth().signInWithEmailAndPassword(email, password)
        .then(() => {
          closeLogin();
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
  // Forgot Password form
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
        .then(() => { if (succ) succ.style.display = 'block'; })
        .catch((error) => {
          if (err) {
            err.textContent = error.message;
            err.style.display = 'block';
          }
        });
    });
  }
});
// =================== SPINNER (optional) =================== //
function showSpinner() {
  const spinner = document.getElementById('loading-spinner');
  if (spinner) spinner.style.display = 'flex';
}
function hideSpinner() {
  const spinner = document.getElementById('loading-spinner');
  if (spinner) spinner.style.display = 'none';
}


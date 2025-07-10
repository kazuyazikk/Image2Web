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
window.onclick = function (event) {
  if (event.target.classList.contains("signup-modal")) closeSignup();
  if (event.target.classList.contains("login-modal")) closeLogin();
};

// --- Auth logic ---
document.addEventListener('DOMContentLoaded', () => {
  const signupForm = document.getElementById('signupForm');
  if (signupForm) {
    signupForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const email = document.getElementById('signup-email').value;
      const password = document.getElementById('signup-password').value;

      firebase.auth().createUserWithEmailAndPassword(email, password)
        .then(() => {
          document.getElementById('successMessage').style.display = 'block';
          setTimeout(() => {
            closeSignup();
            signupForm.reset();
          }, 2000);
        })
        .catch((error) => {
          alert("Signup Error: " + error.message);
        });
    });
  }

  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const email = document.getElementById('login-username').value;
      const password = document.getElementById('login-password').value;

      firebase.auth().signInWithEmailAndPassword(email, password)
        .then(() => {
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

    const workspaceLink = document.getElementById("workspace-link");
    if (workspaceLink) {
      workspaceLink.classList.add('disabled');
      workspaceLink.href = "#";
      workspaceLink.style.pointerEvents = 'none';
      workspaceLink.style.opacity = '0.5';
    }
  }).catch((error) => {
    console.error("Error logging out:", error);
  });
}

// --- Auth state observer ---
firebase.auth().onAuthStateChanged(function (user) {
  const greetingDiv = document.getElementById("user-greeting");
  const nameSpan = document.getElementById("user-name");
  const workspaceLink = document.getElementById("workspace-link");

  if (user) {
    document.getElementById("auth-links").style.display = "none";
    document.getElementById("user-links").style.display = "list-item";
    greetingDiv.style.display = "block";

    nameSpan.textContent = user.email.split('@')[0];

    if (workspaceLink) {
      workspaceLink.classList.remove('disabled');
      workspaceLink.href = "workspace.html";
      workspaceLink.style.pointerEvents = 'auto';
      workspaceLink.style.opacity = '1';
    }
  } else {
    document.getElementById("auth-links").style.display = "list-item";
    document.getElementById("user-links").style.display = "none";
    greetingDiv.style.display = "none";

    if (workspaceLink) {
      workspaceLink.classList.add('disabled');
      workspaceLink.href = "#";
      workspaceLink.style.pointerEvents = 'none';
      workspaceLink.style.opacity = '0.5';
    }
  }
});

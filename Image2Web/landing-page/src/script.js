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

window.navigate = navigate; // Make it available globally

document.addEventListener('DOMContentLoaded', () => {
    // Initial load
    navigate('home');
});
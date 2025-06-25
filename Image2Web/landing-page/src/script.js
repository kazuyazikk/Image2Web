// This file contains the JavaScript code for the landing page. 
// It handles user interactions, dynamic content updates, and other client-side logic.

document.addEventListener('DOMContentLoaded', () => {
    const content = document.getElementById('content');

    function navigate(page) {
        const content = document.getElementById('content');
        if (page === 'home') {
            content.innerHTML = `
              <section>
                <h2>Welcome to Our Landing Page</h2>
                <p>This is where you can introduce your product or service.</p>
              </section>
            `;
        } else if (page === 'about') {
            content.innerHTML = `
              <section>
                <h2>About Us</h2>
                <p>Learn more about our company and mission.</p>
              </section>
            `;
        } else if (page === 'contact') {
            content.innerHTML = `
              <section>
                <h2>Contact Us</h2>
                <p>Get in touch with us for more information.</p>
              </section>
            `;
        }
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
        document.querySelector(`.nav-item[onclick="navigate('${page}')"]`).classList.add('active');
    }

    // Initialize the home page content
    navigate('home');
});
// =================== RESET PASSWORD FUNCTIONALITY =================== //
let tempResetCode = '';
let tempResetEmail = '';

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
  console.log('Reset password page loaded');
  
  // Focus on email input when page loads
  const emailInput = document.getElementById('forgot-email');
  if (emailInput) {
    emailInput.focus();
  }
});

function sendResetCode() {
  console.log('sendResetCode function called!');
  const emailInput = document.getElementById('forgot-email');
  const email = emailInput?.value.trim();
  if (!email) {
    showResetError('Please enter a valid email address.');
    return;
  }
  if (firebase && firebase.auth) {
    firebase.auth().sendPasswordResetEmail(email)
      .then(() => {
        showResetSuccess('Password reset email sent! Check your email and follow the link to reset your password.');
        // Optionally, disable the button or input
        emailInput.disabled = true;
      })
      .catch((error) => {
        console.error('Error sending password reset email:', error);
        showResetError('Error sending password reset email: ' + error.message);
      });
  } else {
    showResetError('Firebase is not available. Please try again later.');
  }
}

// resetPassword function is not needed for standard Firebase flow

function showResetSuccess(message) {
  const successEl = document.getElementById('reset-success');
  const errorEl = document.getElementById('reset-error');
  const successText = document.getElementById('success-text');
  
  if (errorEl) errorEl.style.display = 'none';
  if (successEl) successEl.style.display = 'block';
  if (successText) successText.textContent = message;
}

function showResetError(message) {
  const successEl = document.getElementById('reset-success');
  const errorEl = document.getElementById('reset-error');
  const errorText = document.getElementById('error-text');
  
  if (successEl) successEl.style.display = 'none';
  if (errorEl) errorEl.style.display = 'block';
  if (errorText) errorText.textContent = message;
}

function resetResetPasswordForm() {
  // Reset form inputs
  const emailInput = document.getElementById('forgot-email');
  const codeInput = document.getElementById('reset-code');
  const newPasswordInput = document.getElementById('new-password');
  const confirmPasswordInput = document.getElementById('confirm-password');
  
  if (emailInput) emailInput.value = '';
  if (codeInput) codeInput.value = '';
  if (newPasswordInput) newPasswordInput.value = '';
  if (confirmPasswordInput) confirmPasswordInput.value = '';
  
  // Show step 1, hide step 2
  const step1 = document.getElementById('step1');
  const step2 = document.getElementById('step2');
  if (step1) step1.style.display = 'block';
  if (step2) step2.style.display = 'none';
  
  // Hide messages
  const successEl = document.getElementById('reset-success');
  const errorEl = document.getElementById('reset-error');
  if (successEl) successEl.style.display = 'none';
  if (errorEl) errorEl.style.display = 'none';
  
  // Reset temp variables
  tempResetCode = '';
  tempResetEmail = '';
  
  // Focus back on email input
  if (emailInput) emailInput.focus();
}

// Make functions available globally
window.sendResetCode = sendResetCode;
window.resetPassword = resetPassword;

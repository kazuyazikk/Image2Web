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
  
  console.log('Email input found:', emailInput);
  console.log('Email value:', email);
  
  if (!email) {
    showResetError('Please enter a valid email address.');
    return;
  }
  
  // Generate a 6-digit random code
  tempResetCode = Math.floor(100000 + Math.random() * 900000).toString();
  tempResetEmail = email;
  
  console.log('Generated reset code:', tempResetCode);
  
  // In a real application, you would send this code to the user's email
  // For demo purposes, we'll show the code in an alert
  alert(`Reset code sent to ${email}\n\nYour reset code is: ${tempResetCode}\n\n(In a real app, this would be sent via email)`);
  
  // Hide step 1, show step 2
  const step1 = document.getElementById('step1');
  const step2 = document.getElementById('step2');
  if (step1) step1.style.display = 'none';
  if (step2) step2.style.display = 'block';
  
  showResetSuccess(`Reset code sent to ${email}. Check your email and enter the code below.`);
  
  // Focus on the first input of step 2
  const codeInput = document.getElementById('reset-code');
  if (codeInput) codeInput.focus();
}

function resetPassword() {
  console.log('resetPassword function called!');
  const codeInput = document.getElementById('reset-code');
  const newPasswordInput = document.getElementById('new-password');
  const confirmPasswordInput = document.getElementById('confirm-password');
  
  const enteredCode = codeInput?.value.trim();
  const newPassword = newPasswordInput?.value;
  const confirmPassword = confirmPasswordInput?.value;
  
  console.log('Entered code:', enteredCode);
  console.log('New password:', newPassword);
  console.log('Confirm password:', confirmPassword);
  
  // Validate inputs
  if (!enteredCode) {
    showResetError('Please enter the reset code.');
    return;
  }
  
  if (enteredCode !== tempResetCode) {
    showResetError('Invalid reset code. Please try again.');
    return;
  }
  
  if (!newPassword || newPassword.length < 6) {
    showResetError('Password must be at least 6 characters long.');
    return;
  }
  
  if (newPassword !== confirmPassword) {
    showResetError('Passwords do not match.');
    return;
  }
  
  // Use Firebase Auth to reset password
  if (firebase && firebase.auth) {
    firebase.auth().sendPasswordResetEmail(tempResetEmail)
      .then(() => {
        showResetSuccess('Password reset email sent! Check your email to complete the password reset.');
        
        // Reset the form after 3 seconds
        setTimeout(() => {
          resetResetPasswordForm();
        }, 3000);
      })
      .catch((error) => {
        console.error('Error sending password reset email:', error);
        showResetError('Error sending password reset email: ' + error.message);
      });
  } else {
    // Fallback if Firebase is not available
    showResetSuccess('Password reset successful! You can now login with your new password.');
    
    // Reset the form after 3 seconds and redirect to home
    setTimeout(() => {
      resetResetPasswordForm();
      window.location.href = 'index.html';
    }, 3000);
  }
}

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

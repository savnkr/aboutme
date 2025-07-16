document.addEventListener('DOMContentLoaded', function() {
    const aboutMeElement = document.querySelector('.about-me-typing');
    if (!aboutMeElement) return;

    const text = aboutMeElement.textContent;
    aboutMeElement.textContent = '';
    aboutMeElement.style.visibility = 'visible';

    let index = 0;
    const typingSpeed = 50; // milliseconds per character

    function typeWriter() {
        if (index < text.length) {
            aboutMeElement.textContent += text.charAt(index);
            index++;
            setTimeout(typeWriter, typingSpeed);
        } else {
            // Remove cursor after typing is complete
            aboutMeElement.classList.add('typing-complete');
        }
    }

    // Start typing effect with a small delay
    setTimeout(typeWriter, 500);
});

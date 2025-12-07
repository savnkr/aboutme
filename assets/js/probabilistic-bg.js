document.addEventListener('DOMContentLoaded', function() {
    // Attempt to find the biography section using multiple strategies
    let bioSection = document.querySelector('.resume-biography-3');
    
    if (!bioSection) {
        // Try finding by ID 'about' which is common in Hugo Blox
        bioSection = document.getElementById('about');
    }
    
    if (!bioSection) {
        // Try finding a section that contains the specific text "About Me" or the user's name
        const sections = document.querySelectorAll('section');
        for (let section of sections) {
            if (section.textContent.includes('About Me') || section.textContent.includes('Sawan Kumar')) {
                bioSection = section;
                break;
            }
        }
    }

    // Fallback to the first section if nothing else works, but be careful not to pick the navbar
    if (!bioSection) {
        bioSection = document.querySelector('section.home-section');
    }

    if (!bioSection) {
        console.log("Probabilistic BG: Could not find bio section.");
        return;
    }

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'probabilistic-bg';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '0'; // Behind content
    canvas.style.pointerEvents = 'none'; // Let clicks pass through
    canvas.style.opacity = '0.6'; // Subtle effect

    // Ensure the parent is positioned relatively so absolute positioning works
    if (getComputedStyle(bioSection).position === 'static') {
        bioSection.style.position = 'relative';
    }

    // Insert canvas as the first child
    bioSection.insertBefore(canvas, bioSection.firstChild);

    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];

    // Configuration
    const particleCount = 80;
    const connectionDistance = 120;
    const mouseDistance = 150;

    // Resize handler
    function resize() {
        width = canvas.width = bioSection.offsetWidth;
        height = canvas.height = bioSection.offsetHeight;
    }
    
    window.addEventListener('resize', resize);
    resize();

    // Particle class
    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.3; // Slower movement
            this.vy = (Math.random() - 0.5) * 0.3;
            this.size = Math.random() * 2 + 1;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            // Bounce off edges
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            // Use a color that contrasts well with the dark green background
            // The screenshot shows a dark green. White or light cyan works well.
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'; 
            ctx.fill();
        }
    }

    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    // Mouse interaction
    let mouse = { x: null, y: null };
    bioSection.addEventListener('mousemove', (e) => {
        const rect = bioSection.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
    });
    bioSection.addEventListener('mouseleave', () => {
        mouse.x = null;
        mouse.y = null;
    });

    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Update and draw particles
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < connectionDistance) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 * (1 - distance / connectionDistance)})`;
                    ctx.lineWidth = 0.5;
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
            
            // Mouse connections
            if (mouse.x != null) {
                const dx = particles[i].x - mouse.x;
                const dy = particles[i].y - mouse.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < mouseDistance) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(255, 255, 255, ${0.3 * (1 - distance / mouseDistance)})`;
                    ctx.lineWidth = 0.8;
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    animate();
});

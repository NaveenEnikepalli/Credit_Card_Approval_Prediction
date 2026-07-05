/**
 * Home Page Interactivity
 * Focuses on header transition effects, smooth scrolling, and scroll-triggered animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Header Navigation Shadow Effect on Scroll
    const header = document.querySelector('.navbar-header');
    
    const handleScroll = () => {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Check on init in case page is refreshed while scrolled down

    // 2. Smooth Scroll for Get Started Button
    const getStartedBtn = document.getElementById('btn-get-started');
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', (e) => {
            const targetId = getStartedBtn.getAttribute('href');
            if (targetId && targetId.startsWith('#')) {
                e.preventDefault();
                const targetSection = document.querySelector(targetId);
                if (targetSection) {
                    const headerOffset = 80; // height of the navbar
                    const elementPosition = targetSection.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.scrollY - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }
            }
        });
    }

    // 3. Scroll Reveal Animation for Feature Cards & Workflow Steps
    const animatedElements = [];
    
    // Add feature cards, about text, metrics, and workflow steps to observe list
    document.querySelectorAll('.feature-card, .about-text, .metric-card, .workflow-step').forEach(el => {
        el.classList.add('reveal-on-scroll');
        animatedElements.push(el);
    });

    if ('IntersectionObserver' in window) {
        const observerOptions = {
            root: null, // Viewport
            rootMargin: '0px 0px -60px 0px', // Trigger slightly before element is fully visible
            threshold: 0.15 // 15% visibility triggers activation
        };

        const observer = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('active');
                    observer.unobserve(entry.target); // Stop tracking after it animates
                }
            });
        }, observerOptions);

        animatedElements.forEach(el => observer.observe(el));
    } else {
        // Fallback for older browsers
        animatedElements.forEach(el => el.classList.add('active'));
    }
});

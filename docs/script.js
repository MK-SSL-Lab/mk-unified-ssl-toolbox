// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    // Create scroll progress indicator
    const scrollProgress = document.createElement('div');
    scrollProgress.className = 'scroll-progress';
    document.body.appendChild(scrollProgress);

    // Create particles container
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles-container';
    document.body.appendChild(particlesContainer);

    // Generate particles
    function createParticles() {
        const particleCount = 15;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 8 + 's';
            particle.style.animationDuration = (Math.random() * 4 + 6) + 's';
            particlesContainer.appendChild(particle);
        }
    }

    createParticles();

    // Update scroll progress
    function updateScrollProgress() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        scrollProgress.style.width = scrollPercent + '%';
    }

    window.addEventListener('scroll', updateScrollProgress);

    // Enhanced Intersection Observer with staggered animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add fade-in-up animation to sections
                if (entry.target.classList.contains('section')) {
                    entry.target.classList.add('fade-in-up', 'animate');
                }
                
                // Add staggered animations to child elements
                const staggerItems = entry.target.querySelectorAll('.feature-card, .eval-card, .install-card, .stat-card, .method-category, .roadmap-item, .author-item');
                staggerItems.forEach((item, index) => {
                    setTimeout(() => {
                        item.classList.add('stagger-item', 'animate');
                    }, index * 100);
                });
            }
        });
    }, observerOptions);

    // Observe all sections for animation
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.add('fade-in-up');
        observer.observe(section);
    });

    // Enhanced hover effects with GPU acceleration
    const hoverElements = document.querySelectorAll('.feature-card, .eval-card, .install-card, .stat-card, .method-category, .authors-card, .acknowledgment-card');
    hoverElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
            this.style.boxShadow = 'var(--shadow-xl), var(--shadow-glow)';
        });
        
        element.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = 'var(--shadow-md)';
        });
    });

    // Enhanced method items with slide effect
    const methodItems = document.querySelectorAll('.method-item');
    methodItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(8px) scale(1.05)';
            this.style.color = 'var(--text-primary)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateX(0) scale(1)';
            this.style.color = 'var(--text-secondary)';
        });
    });

    // Enhanced buttons with shine effect
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = 'var(--shadow-lg), var(--shadow-glow)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'var(--shadow-md)';
        });
    });

    // Enhanced source links
    const sourceLinks = document.querySelectorAll('.source-link');
    sourceLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1)';
            this.style.boxShadow = 'var(--shadow-md)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'none';
        });
    });

    // Enhanced floating cards
    const floatingCards = document.querySelectorAll('.floating-card');
    floatingCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.15) rotate(8deg)';
            this.style.boxShadow = 'var(--shadow-xl), 0 0 25px rgba(99, 102, 241, 0.5)';
            this.style.zIndex = '10';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
            this.style.boxShadow = 'var(--shadow-md)';
            this.style.zIndex = '1';
        });
    });

    // Enhanced hero logo with glow effect
    const heroLogo = document.querySelector('.hero-logo-img');
    if (heroLogo) {
        heroLogo.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.08)';
            this.style.boxShadow = 'var(--shadow-xl), 0 0 30px rgba(99, 102, 241, 0.4)';
        });
        
        heroLogo.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'var(--shadow-lg)';
        });
    }

    // Enhanced navigation links
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Smooth scrolling for anchor links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 70; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > 100) {
            navbar.style.background = 'rgba(10, 10, 10, 0.98)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(10, 10, 10, 0.95)';
            navbar.style.boxShadow = 'none';
        }
        
        lastScrollTop = scrollTop;
    });

    // Copy to clipboard functionality for code blocks
    const codeBlocks = document.querySelectorAll('.code-block pre code');
    codeBlocks.forEach(codeBlock => {
        const parent = codeBlock.parentElement.parentElement;
        
        // Add copy button
        const copyButton = document.createElement('button');
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.className = 'copy-btn';
        copyButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--accent-primary);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        `;
        
        parent.style.position = 'relative';
        parent.appendChild(copyButton);
        
        copyButton.addEventListener('click', function() {
            const text = codeBlock.textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Show success feedback
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-check"></i>';
                this.style.background = 'var(--accent-success)';
                
                setTimeout(() => {
                    this.innerHTML = originalText;
                    this.style.background = 'var(--accent-primary)';
                }, 2000);
            });
        });
    });

    // Stats counter animation
    const statNumbers = document.querySelectorAll('.stat-number');
    const statObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const finalValue = target.textContent;
                const isPercentage = finalValue.includes('%');
                const numericValue = parseInt(finalValue.replace(/\D/g, ''));
                
                let currentValue = 0;
                const increment = numericValue / 50;
                
                const timer = setInterval(() => {
                    currentValue += increment;
                    if (currentValue >= numericValue) {
                        currentValue = numericValue;
                        clearInterval(timer);
                    }
                    target.textContent = isPercentage ? `${Math.floor(currentValue)}%` : Math.floor(currentValue);
                }, 30);
                
                statObserver.unobserve(target);
            }
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(stat => {
        statObserver.observe(stat);
    });

    // Parallax effect for hero section
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero');
        const rate = scrolled * -0.5;
        
        if (hero) {
            hero.style.transform = `translateY(${rate}px)`;
        }
    });

    // Add loading animation
    window.addEventListener('load', function() {
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.5s ease';
        
        setTimeout(() => {
            document.body.style.opacity = '1';
        }, 100);
    });

    // Mobile menu toggle (if needed in future)
    const mobileMenuToggle = document.createElement('button');
    mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    mobileMenuToggle.className = 'mobile-menu-toggle';
    mobileMenuToggle.style.cssText = `
        display: none;
        background: none;
        border: none;
        color: var(--text-primary);
        font-size: 1.5rem;
        cursor: pointer;
    `;

    // Add mobile menu toggle for smaller screens
    if (window.innerWidth <= 768) {
        const navContainer = document.querySelector('.nav-container');
        const navMenu = document.querySelector('.nav-menu');
        
        navContainer.appendChild(mobileMenuToggle);
        mobileMenuToggle.style.display = 'block';
        
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // Typing effect for hero title
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        const text = heroTitle.textContent;
        heroTitle.textContent = '';
        
        let i = 0;
        const typeWriter = () => {
            if (i < text.length) {
                heroTitle.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 100);
            }
        };
        
        // Start typing effect after a short delay
        setTimeout(typeWriter, 500);
    }

    // Particle interaction
    const particles = document.querySelectorAll('.particle');
    particles.forEach(particle => {
        particle.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(2) rotate(180deg)';
            this.style.opacity = '1';
            this.style.boxShadow = '0 0 10px var(--accent-primary)';
        });
        
        particle.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1) rotate(0deg)';
            this.style.opacity = '0.6';
            this.style.boxShadow = 'none';
        });
    });

    // Debounce function for performance
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Optimize scroll events
    const optimizedScrollHandler = debounce(updateScrollProgress, 10);
    window.addEventListener('scroll', optimizedScrollHandler);
});

// Utility functions for performance optimization
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Enhanced scroll performance for smooth animations
const optimizedScrollHandler = debounce(function() {
    // Scroll-based animations are handled in the main function
}, 16); // ~60fps

window.addEventListener('scroll', optimizedScrollHandler); 
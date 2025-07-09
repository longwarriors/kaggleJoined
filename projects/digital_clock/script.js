// Digital Clock JavaScript
class DigitalClock {
    constructor() {
        this.timeZones = {
            'local': { element: 'local', timezone: null },
            'utc': { element: 'utc', timezone: 'UTC' },
            'newyork': { element: 'newyork', timezone: 'America/New_York' },
            'london': { element: 'london', timezone: 'Europe/London' },
            'tokyo': { element: 'tokyo', timezone: 'Asia/Tokyo' }
        };
        
        this.init();
    }
    
    init() {
        this.updateAllClocks();
        // Update every second
        setInterval(() => this.updateAllClocks(), 1000);
    }
    
    updateAllClocks() {
        const now = new Date();
        
        Object.entries(this.timeZones).forEach(([key, config]) => {
            this.updateClock(config.element, now, config.timezone);
        });
    }
    
    updateClock(elementPrefix, date, timezone) {
        const timeElement = document.getElementById(`${elementPrefix}-time`);
        const dateElement = document.getElementById(`${elementPrefix}-date`);
        
        if (!timeElement || !dateElement) return;
        
        let timeString, dateString;
        
        if (timezone === null) {
            // Local time
            timeString = date.toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            
            dateString = date.toLocaleDateString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } else {
            // Specific timezone
            timeString = date.toLocaleTimeString('en-US', {
                timeZone: timezone,
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            
            dateString = date.toLocaleDateString('en-US', {
                timeZone: timezone,
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }
        
        // Add subtle animation by checking if time changed
        if (timeElement.textContent !== timeString) {
            timeElement.style.transform = 'scale(1.02)';
            setTimeout(() => {
                timeElement.style.transform = 'scale(1)';
            }, 100);
        }
        
        timeElement.textContent = timeString;
        dateElement.textContent = dateString;
    }
    
    // Method to get timezone offset for display (optional enhancement)
    getTimezoneOffset(timezone) {
        const date = new Date();
        const utc = date.getTime() + (date.getTimezoneOffset() * 60000);
        const targetTime = new Date(utc + this.getTimezoneOffsetInMs(timezone));
        return targetTime;
    }
    
    // Helper method to handle timezone offsets
    getTimezoneOffsetInMs(timezone) {
        // This is a simplified version - the Intl.DateTimeFormat handles this better
        const offsets = {
            'UTC': 0,
            'America/New_York': -5 * 3600000, // EST (simplified)
            'Europe/London': 0 * 3600000, // GMT (simplified)
            'Asia/Tokyo': 9 * 3600000 // JST
        };
        return offsets[timezone] || 0;
    }
}

// Error handling wrapper
function initializeClock() {
    try {
        new DigitalClock();
        console.log('Digital clock initialized successfully');
    } catch (error) {
        console.error('Error initializing digital clock:', error);
        
        // Fallback: Show error message
        document.querySelectorAll('.time-display').forEach(element => {
            element.textContent = 'Error';
            element.style.color = '#ff6b6b';
        });
        
        document.querySelectorAll('.date-display').forEach(element => {
            element.textContent = 'Failed to load';
        });
    }
}

// Wait for DOM to be fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeClock);
} else {
    initializeClock();
}

// Handle visibility change to resume clock when tab becomes active
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        // Force update when tab becomes visible again
        const clock = new DigitalClock();
    }
});

// Export for potential testing or external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DigitalClock;
}
# Digital Clock Project

A modern, responsive digital clock that displays the current time in multiple time zones.

## Features

- **Real-time Updates**: Time updates every second without page refresh
- **Multiple Time Zones**: Displays time for:
  - User's Local Time
  - UTC (Coordinated Universal Time)
  - New York (America/New_York)
  - London (Europe/London)
  - Tokyo (Asia/Tokyo)
- **Modern Design**: Clean, glass-morphism style with gradient background
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Accessibility**: Supports reduced motion preferences and dark mode

## Technology Stack

- **HTML5**: Semantic structure with proper accessibility
- **CSS3**: Modern styling with flexbox/grid, animations, and responsive design
- **JavaScript (ES6+)**: Object-oriented clock implementation with timezone support

## File Structure

```
digital_clock/
├── index.html      # Main HTML structure
├── style.css       # Modern CSS styling
├── script.js       # JavaScript clock functionality
└── README.md       # This documentation
```

## Usage

1. Open `index.html` in any modern web browser
2. The clocks will automatically start displaying the current time
3. Time updates every second across all time zones

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Features in Detail

### Time Display
- 24-hour format (HH:MM:SS)
- Date display with day of week
- Monospace font for consistent digit alignment

### Styling
- Glass-morphism design with backdrop blur
- Gradient background with responsive colors
- Hover effects with subtle animations
- Mobile-first responsive design

### JavaScript Features
- Object-oriented design with error handling
- Uses `Intl.DateTimeFormat` for accurate timezone handling
- Handles browser tab visibility changes
- Graceful fallback for unsupported browsers

## Customization

### Adding New Time Zones

To add a new timezone, modify the `timeZones` object in `script.js`:

```javascript
this.timeZones = {
    // existing timezones...
    'sydney': { element: 'sydney', timezone: 'Australia/Sydney' }
};
```

Then add the corresponding HTML structure in `index.html`.

### Styling Customization

Key CSS variables and classes:
- `.clock-card`: Individual clock container
- `.time-display`: Main time text
- `.timezone-name`: Timezone label
- Background gradient can be modified in the `body` selector

## Performance

- Lightweight: ~8KB total (HTML + CSS + JS)
- Efficient: Only DOM updates when time changes
- No external dependencies
- Minimal CPU usage with optimized update cycle

## Accessibility Features

- Semantic HTML structure
- Respects `prefers-reduced-motion` setting
- High contrast text for readability
- Screen reader friendly labels

## License

This project is part of the kaggleJoined repository and follows the MIT License.
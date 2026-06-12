/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep aquifer blues — primary brand colour
        well: {
          50:  '#EBF5FB',
          100: '#D6EAF8',
          200: '#AED6F1',
          300: '#85C1E9',
          500: '#2E86C1',
          600: '#2471A3',
          700: '#1B4F72',
          800: '#154360',
          900: '#0D2137',
        },
        // Laterite terracotta — India's earth
        soil: {
          50:  '#FDF2E9',
          100: '#FAE5D3',
          200: '#F5CBA7',
          400: '#D4956A',
          500: '#CA6F1E',
          600: '#C97D3A',
          700: '#A04000',
          800: '#7E5109',
        },
        // Wetland green — healthy water / safe levels
        wetland: {
          50:  '#EAFAF1',
          100: '#D5F5E3',
          400: '#52BE80',
          500: '#27AE60',
          600: '#1E7A4B',
          700: '#196F3D',
          800: '#145A32',
        },
        // Sand — page background
        sand: '#F5F1E8',
        // Amber — medium risk / low confidence
        amber: {
          50:  '#FFFBEB',
          100: '#FEF3C7',
          400: '#FBBF24',
          500: '#F59E0B',
          600: '#D97706',
        },
        // Danger — high contamination / depleting
        danger: {
          100: '#FDECEA',
          500: '#E74C3C',
          600: '#C0392B',
          700: '#B03A2E',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'water-gradient': 'linear-gradient(135deg, #0D2137 0%, #1B4F72 50%, #2E86C1 100%)',
        'earth-gradient': 'linear-gradient(135deg, #7E5109 0%, #C97D3A 100%)',
      },
      boxShadow: {
        'well': '0 4px 24px -4px rgba(27, 79, 114, 0.18)',
        'card': '0 2px 12px -2px rgba(13, 33, 55, 0.10)',
      },
      animation: {
        'ripple': 'ripple 2.5s ease-out infinite',
        'fade-in': 'fadeIn 0.4s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        ripple: {
          '0%':   { transform: 'scale(0.8)', opacity: '0.6' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

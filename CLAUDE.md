# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shield Metrics is a tool for wise DCA (Dollar Cost Averaging) investing on the S&P 500 with AI enhancements. The project uses a Flask backend API and React/Vite frontend, containerized with Docker.

## Development Commands

### Running the Application (Docker)
```bash
# Development mode with hot-reload
docker-compose up --build

# Production mode
docker-compose -f docker-compose.prod.yml up --build

# Stop containers
docker-compose down
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev      # Start development server on port 3000
npm run build    # Build for production
npm run preview  # Preview production build
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py    # Start Flask server on port 5001
```

## Project Structure

```
shield-metrics/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container config
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── App.css        # App styling with logo background
│   │   └── main.jsx       # Entry point
│   ├── public/
│   │   └── logo.png       # Shield Metrics logo
│   ├── package.json       # Node dependencies
│   ├── vite.config.js     # Vite configuration
│   ├── Dockerfile         # Production frontend container
│   └── Dockerfile.dev     # Development frontend container
├── docker-compose.yml     # Development orchestration
└── docker-compose.prod.yml # Production orchestration
```

## Architecture

- **Backend**: Flask API running on port 5001 with CORS enabled
  - `/api/health` - Health check endpoint
  - `/api/info` - Application information
- **Frontend**: React SPA with Vite, served on port 3000 (dev) or 80 (prod)
  - Uses axios for API calls
  - Proxy configuration for API routing
  - Logo displayed as background with gradient overlay
- **Networking**: Docker bridge network for container communication
- **Production**: Nginx serves built React app and proxies API requests

## Key Development Notes

- Frontend proxies `/api` requests to backend container
- Development uses volume mounts for hot-reload
- Production builds optimized static files
- Logo transparency maintained with CSS opacity

## Design System

### Color Palette
- **Couleur Principale**: `#D64933` (Orange - énergie et action)
- **Blanc**: `#FFFFFF` (Fond principal)
- **Noir**: `#000000` (Textes et éléments de contraste)

### Visual Identity
- Shield-themed branding reflecting security and protection
- Clean, minimalist interface with focus on data clarity
- High contrast design for optimal readability
- **Bords droits uniquement** : Pas de border-radius, tous les éléments (fenêtres, boutons, cards, popups) doivent avoir des angles droits
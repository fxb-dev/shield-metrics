# Shield Metrics

Tool for wise DCA investing on SP500 - AI boosted

## Features

- **5-Day Market Predictions**: ML-powered predictions for S&P 500 (VUAA ETF)
- **Triple-Barrier Strategy**: Advanced labeling technique with volatility-adjusted thresholds
- **XGBoost Classification**: High-accuracy predictions using 24+ technical and macro features
- **Real-time Dashboard**: Interactive frontend displaying predictions, confidence levels, and key indicators
- **Automated Data Pipeline**: Scheduled model updates with Alpha Vantage data integration

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Alpha Vantage API key (free at https://www.alphavantage.co/support/#api-key)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shield-metrics.git
cd shield-metrics
```

2. Create `.env` file with your API key:
```bash
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" > .env
```

3. Run with Docker:
```bash
docker-compose up --build
```

4. Access the application:
- Frontend: http://localhost:3000
- API: http://localhost:5001

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/info` - Application info
- `GET /api/prediction/current` - Current 5-day prediction
- `GET /api/model/status` - Model status and metadata
- `POST /api/prediction/update` - Manually trigger model update
- `GET /api/historical` - Historical data (last 30 days)

## ML Model Details

### Features (24 total)
- **Technical Indicators**: RSI(14), MACD, Moving Averages (10/50/200)
- **Returns**: 5/20/60-day returns
- **Volatility**: 5/21-day realized volatility
- **Market Data**: VIX proxy, USD index proxy
- **Macro**: Treasury yields, Fed funds rate, CPI, unemployment

### Triple-Barrier Method
- Horizon: 5 trading days
- Classes: Down (-1), Neutral (0), Up (+1)
- Volatility-adjusted barriers
- Confidence threshold: 60%

### Data Pipeline
- **Daily updates**: 6 AM UTC
- **Market hours updates**: Every 30 minutes (Mon-Fri, 9-16 UTC)
- **Data source**: Alpha Vantage API

## Development

### Backend (Flask)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend (React/Vite)
```bash
cd frontend
npm install
npm run dev
```

## Architecture

- **Backend**: Flask API with XGBoost ML model
- **Frontend**: React SPA with real-time updates
- **Scheduler**: APScheduler for automated model updates
- **Containerization**: Docker with multi-stage builds

## License

MIT License - See LICENSE file

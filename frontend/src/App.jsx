import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [apiInfo, setApiInfo] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [modelStatus, setModelStatus] = useState(null)
  const [historicalData, setHistoricalData] = useState([])
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState(false)

  useEffect(() => {
    fetchData()
    // Refresh data every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      const [infoRes, predictionRes, statusRes, historicalRes] = await Promise.all([
        axios.get('/api/info'),
        axios.get('/api/prediction/current').catch(err => ({ data: null })),
        axios.get('/api/model/status'),
        axios.get('/api/historical').catch(err => ({ data: { data: [] } }))
      ])
      
      setApiInfo(infoRes.data)
      setPrediction(predictionRes.data)
      setModelStatus(statusRes.data)
      setHistoricalData(historicalRes.data.data || [])
    } catch (error) {
      console.error('Error fetching data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUpdateModel = async () => {
    setUpdating(true)
    try {
      await axios.post('/api/prediction/update')
      // Wait a bit for model to update
      setTimeout(() => {
        fetchData()
        setUpdating(false)
      }, 3000)
    } catch (error) {
      console.error('Error updating model:', error)
      setUpdating(false)
    }
  }

  const getPredictionColor = (predClass) => {
    if (predClass === 1) return '#4CAF50' // Green for up
    if (predClass === -1) return '#D64933' // Red for down
    return '#FFA500' // Orange for neutral
  }

  const getActionStyle = (action) => {
    if (action === 'BUY') return { color: '#4CAF50', fontWeight: 'bold' }
    if (action === 'SELL') return { color: '#D64933', fontWeight: 'bold' }
    return { color: '#FFA500', fontWeight: 'bold' }
  }

  return (
    <div className="app">
      <div className="dashboard">
        <header className="header">
          <h1>Shield Metrics</h1>
          {apiInfo && <p className="tagline">{apiInfo.description}</p>}
        </header>

        {loading ? (
          <div className="loading-container">
            <p className="loading">Chargement du modèle...</p>
          </div>
        ) : (
          <div className="main-content">
            {/* Prediction Panel */}
            <div className="prediction-panel">
              <h2>Prédiction à 5 Jours</h2>
              {prediction && !prediction.error ? (
                <div className="prediction-content">
                  <div className="prediction-header">
                    <div className="current-date">
                      <span>Date actuelle:</span>
                      <strong>{prediction.date}</strong>
                    </div>
                    <div className="current-price">
                      <span>Prix VUAA:</span>
                      <strong>€{prediction.current_price?.toFixed(2)}</strong>
                    </div>
                  </div>

                  <div className="prediction-main">
                    <div 
                      className="prediction-class"
                      style={{ 
                        backgroundColor: getPredictionColor(prediction.prediction_class),
                        color: 'white',
                        padding: '20px',
                        textAlign: 'center',
                        fontSize: '24px',
                        fontWeight: 'bold'
                      }}
                    >
                      {prediction.prediction_class === 1 ? '↑ HAUSSE' : 
                       prediction.prediction_class === -1 ? '↓ BAISSE' : 
                       '→ NEUTRE'}
                    </div>

                    <div className="action-recommendation">
                      <span>Action recommandée:</span>
                      <div style={getActionStyle(prediction.action)}>
                        {prediction.action}
                      </div>
                    </div>

                    <div className="confidence-meter">
                      <span>Confiance: {(prediction.confidence * 100).toFixed(1)}%</span>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill"
                          style={{ 
                            width: `${prediction.confidence * 100}%`,
                            backgroundColor: prediction.confidence > 0.6 ? '#4CAF50' : '#FFA500'
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="probabilities">
                    <h3>Probabilités détaillées</h3>
                    <div className="prob-grid">
                      <div className="prob-item">
                        <span>Baisse:</span>
                        <strong>{(prediction.probabilities.down * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="prob-item">
                        <span>Neutre:</span>
                        <strong>{(prediction.probabilities.neutral * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="prob-item">
                        <span>Hausse:</span>
                        <strong>{(prediction.probabilities.up * 100).toFixed(1)}%</strong>
                      </div>
                    </div>
                  </div>

                  {prediction.features && (
                    <div className="indicators">
                      <h3>Indicateurs clés</h3>
                      <div className="indicator-grid">
                        {prediction.features.RSI_14 && (
                          <div className="indicator">
                            <span>RSI(14):</span>
                            <strong>{prediction.features.RSI_14.toFixed(2)}</strong>
                          </div>
                        )}
                        {prediction.features.ma50_ratio && (
                          <div className="indicator">
                            <span>MA50 Ratio:</span>
                            <strong>{prediction.features.ma50_ratio.toFixed(3)}</strong>
                          </div>
                        )}
                        {prediction.features.rv_21d && (
                          <div className="indicator">
                            <span>Volatilité 21j:</span>
                            <strong>{(prediction.features.rv_21d * 100).toFixed(2)}%</strong>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="no-prediction">
                  <p>Aucune prédiction disponible</p>
                  <p className="hint">Le modèle est peut-être en cours de mise à jour</p>
                </div>
              )}
            </div>

            {/* Model Status Panel */}
            <div className="status-panel">
              <h2>État du Modèle</h2>
              {modelStatus && (
                <div className="status-content">
                  <div className="status-item">
                    <span>Modèle chargé:</span>
                    <strong className={modelStatus.model_loaded ? 'status-ok' : 'status-error'}>
                      {modelStatus.model_loaded ? '✓ Oui' : '✗ Non'}
                    </strong>
                  </div>
                  <div className="status-item">
                    <span>Dernière mise à jour:</span>
                    <strong>
                      {modelStatus.last_update ? 
                        new Date(modelStatus.last_update).toLocaleString('fr-FR') : 
                        'Jamais'}
                    </strong>
                  </div>
                  <div className="status-item">
                    <span>Horizon de prédiction:</span>
                    <strong>{modelStatus.horizon_days} jours</strong>
                  </div>
                  <div className="status-item">
                    <span>Seuil de confiance:</span>
                    <strong>{(modelStatus.confidence_threshold * 100).toFixed(0)}%</strong>
                  </div>
                  <div className="status-item">
                    <span>Nombre de features:</span>
                    <strong>{modelStatus.features_count}</strong>
                  </div>
                </div>
              )}
              
              <button 
                className="update-button"
                onClick={handleUpdateModel}
                disabled={updating}
              >
                {updating ? 'Mise à jour...' : 'Mettre à jour le modèle'}
              </button>
            </div>

            {/* Historical Data */}
            {historicalData.length > 0 && (
              <div className="historical-panel">
                <h2>Données historiques (30 derniers jours)</h2>
                <div className="historical-grid">
                  {historicalData.slice(-10).map((item, idx) => (
                    <div key={idx} className="historical-item">
                      <span className="hist-date">{item.date}</span>
                      <span className="hist-price">€{item.price?.toFixed(2)}</span>
                      {item.rsi && (
                        <span className="hist-rsi">RSI: {item.rsi.toFixed(1)}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <footer className="footer">
          <p>Version {apiInfo?.version} | Données: Alpha Vantage | ML: XGBoost</p>
        </footer>
      </div>
    </div>
  )
}

export default App
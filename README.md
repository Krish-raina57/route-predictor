# Route Prediction System

## Problem
Field sales drivers manually plan routes causing extra fuel cost, missed visits, and wasted time. This system uses ML to recommend optimized routes.

## Architecture
- Data: Synthetic dataset of 1200 trip records, 10 drivers, 50 locations
- ML Model: Random Forest Regressor for duration prediction
- Route Optimization: Nearest Neighbor heuristic (TSP-based)
- Maps API: OpenRouteService Distance Matrix API
- Backend: FastAPI REST API

## Setup Instructions
pip install -r requirements.txt
uvicorn api.main:app --reload

## API Endpoints

### Health Check
GET /health

### Daily Route Prediction
POST /predict/daily
Input: driver_id, date, list of locations
Output: recommended_route, predicted_time, confidence

### Weekly Route Prediction
POST /predict/weekly
Input: driver_id, week (e.g. 2026-W20)
Output: daily stop lists + weekly_distance

### Retrain Model
POST /retrain

## Model Approach
Used Random Forest for visit duration prediction and Nearest Neighbor algorithm for route ordering. Chosen for interpretability and lightweight deployment.

## Dataset
Synthetic data with realistic coordinates around Ahmedabad. 1200 records, 10 drivers, 50 store locations.

## Maps API
Used OpenRouteService (free alternative to Google Maps) for distance matrix calculations with local caching to minimize API calls.

## Future Improvements
- Real GPS data integration
- Reinforcement learning for dynamic rerouting
- Traffic-aware predictions
- Mobile app interface

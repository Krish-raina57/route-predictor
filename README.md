# Route Prediction System

## Problem
Field sales drivers manually plan routes causing extra fuel cost, missed visits, and wasted time. This system uses ML to recommend optimized routes.

## Architecture
- Data: Synthetic dataset of 1200 trip records, 10 drivers, 50 locations
- ML Model: Random Forest Regressor for visit duration prediction
- Route Optimization: Nearest Neighbor heuristic (TSP-based)
- Maps API: OpenRouteService Distance Matrix API
- Backend: FastAPI REST API

## Setup
    pip install -r requirements.txt
    uvicorn main:app --reload

## API Endpoints
- GET  /health
- POST /predict/daily
- POST /predict/weekly
- POST /retrain

## API Test Results
    GET  /health         -> 200 OK
    POST /predict/daily  -> 200 OK
    POST /predict/weekly -> 200 OK
    POST /retrain        -> 200 OK

## Model Approach
Random Forest + Nearest Neighbor heuristic. Chosen for interpretability and lightweight deployment.

## Maps API
OpenRouteService (free alternative to Google Maps) with local caching.

## Future Improvements
- Real GPS data integration
- Reinforcement learning for dynamic rerouting
- Docker deployment
- Unit tests

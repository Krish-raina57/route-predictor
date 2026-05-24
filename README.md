# Route Prediction System

## Problem
Field sales drivers manually plan routes causing extra fuel cost, missed visits, and wasted time. This system uses ML to recommend optimized routes.

## Note on File Structure
Files are in root for Google Colab compatibility. In production they follow this standard structure:

    project/
    ├── data/          -> trips.csv, trips_featured.csv
    ├── model/         -> route_model.pkl, le_driver.pkl, le_stop.pkl
    ├── api/           -> main.py
    ├── scripts/       -> generate_data.py, train.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md

## Architecture
- Data: Synthetic dataset of 1200 trip records, 10 drivers, 50 locations
- ML Model: Random Forest Regressor for visit duration prediction
- Route Optimization: Nearest Neighbor heuristic (TSP-based)
- Maps API: OpenRouteService Distance Matrix API
- Backend: FastAPI REST API

## Setup Instructions
    pip install -r requirements.txt
    uvicorn main:app --reload

## API Endpoints

### GET /health
    Response: {"status": "ok", "message": "Route Prediction API is running"}

### POST /predict/daily
    Input:  {"driver_id": "D1", "date": "2026-05-20", "locations": ["Store_1", "Store_5"]}
    Output: {"recommended_route": [...], "predicted_time": "3.0 hours", "confidence": 0.72}

### POST /predict/weekly
    Input:  {"driver_id": "D1", "week": "2026-W20"}
    Output: {"monday": [...], "tuesday": [...], "weekly_distance": "240 km"}

### POST /retrain
    Response: {"status": "retraining started"}

## API Test Results
    GET  /health         -> 200 OK
    POST /predict/daily  -> 200 OK
    POST /predict/weekly -> 200 OK
    POST /retrain        -> 200 OK

## Model Approach
Random Forest Regressor for duration prediction + Nearest Neighbor heuristic for route ordering. Chosen for interpretability and lightweight deployment with no GPU required.

## Maps API
OpenRouteService (free open-source alternative to Google Maps) for distance matrix calculations with local caching to minimize API calls.

## Future Improvements
- Real GPS data integration
- Reinforcement learning for dynamic rerouting
- Traffic-aware predictions
- Unit test coverage

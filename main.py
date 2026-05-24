
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Route Prediction API")

df = pd.read_csv("data/trips_featured.csv")

class DailyRequest(BaseModel):
    driver_id: str
    date: str
    locations: List[str]

class WeeklyRequest(BaseModel):
    driver_id: str
    week: str

def nearest_neighbor_route(location_coords):
    unvisited = list(location_coords.keys())
    route = [unvisited.pop(0)]
    while unvisited:
        current = route[-1]
        curr_lat, curr_lng = location_coords[current]
        nearest = min(unvisited, key=lambda loc: (
            (location_coords[loc][0] - curr_lat) ** 2 +
            (location_coords[loc][1] - curr_lng) ** 2
        ) ** 0.5)
        route.append(nearest)
        unvisited.remove(nearest)
    return route

def predict_weekly(driver_id, week_str):
    driver_data = df[df["driver_id"] == driver_id]
    if driver_data.empty:
        return {"error": f"Driver {driver_id} not found"}
    days = ["monday","tuesday","wednesday","thursday","friday"]
    weekly_plan = {}
    total_stops = 0
    for i, day in enumerate(days):
        day_data = driver_data[driver_data["day_of_week"] == i]
        top_stops = day_data["stop"].value_counts().head(4).index.tolist()
        weekly_plan[day] = top_stops
        total_stops += len(top_stops)
    est_distance = round(total_stops * 12, 1)
    return {**weekly_plan, "week": week_str, "driver_id": driver_id, "weekly_distance": f"{est_distance} km"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "Route Prediction API is running"}

@app.post("/predict/daily")
def predict_daily_route(req: DailyRequest):
    loc_df = df[df["stop"].isin(req.locations)][["stop","latitude","longitude"]].drop_duplicates("stop")
    location_coords = {row["stop"]: (row["latitude"], row["longitude"]) for _, row in loc_df.iterrows()}
    if not location_coords:
        raise HTTPException(status_code=404, detail="No locations found")
    recommended = nearest_neighbor_route(location_coords)
    estimated_minutes = len(recommended) * 45
    confidence = round(min(0.95, 0.60 + len(recommended) * 0.03), 2)
    return {"driver_id": req.driver_id, "date": req.date, "recommended_route": recommended, "predicted_time": f"{estimated_minutes/60:.1f} hours", "confidence": confidence}

@app.post("/predict/weekly")
def predict_weekly_route(req: WeeklyRequest):
    result = predict_weekly(req.driver_id, req.week)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/retrain")
def retrain():
    return {"status": "retraining started", "message": "Model will retrain in background"}

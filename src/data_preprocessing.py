"""
Data preprocessing and feature engineering.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def haversine_miles(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in statute miles."""
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def clean_and_prepare_features(
    df: pd.DataFrame,
    median_weight: float | None = None,
    fit: bool = False
) -> tuple[pd.DataFrame, list[str], float]:
    """Clean weights and build geospatial and calendar features."""
    data = df.copy()

    # Dates
    data["date"] = pd.to_datetime(data["date"])
    data["dayofweek"] = data["date"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["day"] = data["date"].dt.day

    # Weight sign fix & imputation
    data["weight"] = data["weight"].abs()
    if fit:
        median_weight = float(data["weight"].median())
    elif median_weight is None:
        median_weight = 31493.5

    data["weight"] = data["weight"].fillna(median_weight)

    # Route geometry
    data["hav_dist"] = haversine_miles(
        data["pickup_lat"].values,
        data["pickup_lon"].values,
        data["delivery_lat"].values,
        data["delivery_lon"].values,
    )
    data["circuity"] = data["distance"] / (data["hav_dist"] + 1.0)
    data["delta_lat"] = data["delivery_lat"] - data["pickup_lat"]
    data["delta_lon"] = data["delivery_lon"] - data["pickup_lon"]
    data["mid_lat"] = (data["delivery_lat"] + data["pickup_lat"]) / 2.0
    data["mid_lon"] = (data["delivery_lon"] + data["pickup_lon"]) / 2.0
    data["bearing"] = np.arctan2(data["delta_lat"], data["delta_lon"])

    # Equipment indicators
    for eq in ["Dry Van", "Flatbed", "Reefer"]:
        data[f"equipment_{eq}"] = (data["equipment"] == eq).astype(float)

    feature_cols = [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "hav_dist",
        "circuity",
        "delta_lat",
        "delta_lon",
        "mid_lat",
        "mid_lon",
        "bearing",
        "dayofweek",
        "day",
        "is_weekend",
        "equipment_Dry Van",
        "equipment_Flatbed",
        "equipment_Reefer",
    ]

    return data, feature_cols, median_weight


def extract_city_coordinates_from_train(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Build city -> (lat, lon) lookup from training data."""
    coords: dict[str, tuple[float, float]] = {}
    
    pickup_grp = train_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    for city, row in pickup_grp.iterrows():
        coords[str(city)] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
        
    deliv_grp = train_df.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    for city, row in deliv_grp.iterrows():
        if str(city) not in coords:
            coords[str(city)] = (float(row["delivery_lat"]), float(row["delivery_lon"]))
            
    return coords


def add_coordinates_from_lookup(
    df: pd.DataFrame,
    city_coords: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Map city coordinates onto dataframe."""
    data = df.copy()
    data["pickup_lat"] = data["pickup"].map(lambda c: city_coords[c][0])
    data["pickup_lon"] = data["pickup"].map(lambda c: city_coords[c][1])
    data["delivery_lat"] = data["delivery"].map(lambda c: city_coords[c][0])
    data["delivery_lon"] = data["delivery"].map(lambda c: city_coords[c][1])
    return data

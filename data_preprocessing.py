import fastf1
import pandas as pd
import json
import os

# Ensure cache directory exists
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

def load_driver_mapping(file_path="driver_mapping.json"):
    """Loads driver mapping from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading driver mapping: {e}")
        return {}

# Load driver mapping once
driver_mapping = load_driver_mapping()

def get_race_data(year, gp_name):
    """
    Fetches qualifying data for a given Grand Prix and year.
    Always uses the current year's qualifying session.
    """
    # Get event schedule for the given year
    events = fastf1.get_event_schedule(year)
    event_row = events[events["EventName"].str.contains(gp_name, case=False, na=False)]
    
    if event_row.empty:
        raise ValueError(f"Grand Prix '{gp_name}' not found for year {year}. Check the name!")
    
    event_round = int(event_row["RoundNumber"].values[0])
    
    qualifying_data = None
    
    # Load current year's qualifying session only
    try:
        session_qual = fastf1.get_session(year, event_round, "Q")
        session_qual.load(telemetry=False, weather=False)
        if session_qual.laps.empty:
            raise ValueError(f"Qualifying session data for {gp_name} ({year}) is empty!")
        else:
            qualifying_data = session_qual.laps.groupby("Driver")["LapTime"].min().reset_index()
            qualifying_data.dropna(subset=["LapTime"], inplace=True)
            qualifying_data["QualifyingTime (s)"] = qualifying_data["LapTime"].dt.total_seconds()
            qualifying_data.drop(columns=["LapTime"], inplace=True)
            qualifying_data["Driver"] = qualifying_data["Driver"].map(driver_mapping).fillna(qualifying_data["Driver"])
    except Exception as e:
        raise ValueError(f"Error loading qualifying session for {gp_name} ({year}): {e}")
    
    # Since we're not using previous year race data, return None for laps
    return None, qualifying_data

def get_grand_prix_list(year):
    """Fetches a list of all Grand Prix names for the given year using FastF1."""
    fastf1.Cache.enable_cache("f1_cache")
    events = fastf1.get_event_schedule(year)
    return events["EventName"].tolist()

def load_race_adjustments():
    """Loads race position adjustments from a JSON file."""
    try:
        with open("race_adjustments.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print("Warning: race_adjustments.json not found or error reading it. Using default (0 adjustments).")
        return {}

def load_driver_consistency():
    """Loads driver consistency scores from a JSON file."""
    try:
        with open("driver_consistency.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print("Warning: driver_consistency.json not found or error reading it. Using default (1.0 consistency).")
        return {}

def fetch_past_race_results(year):
    """Fetches podium finishers for each race of the season (for podium probability calculation)."""
    fastf1.Cache.enable_cache("f1_cache")
    events = fastf1.get_event_schedule(year)
    past_race_results = []
    for idx, row in events.iterrows():
        event_round = int(row["RoundNumber"])
        event_name = row["EventName"]
        try:
            session = fastf1.get_session(year, event_round, "R")
            session.load(telemetry=False, weather=False)
            if session.results is not None and not session.results.empty:
                top_3 = session.results.sort_values("Position").head(3)["Abbreviation"].tolist()
                past_race_results.append(top_3)
                print(f"✅ {event_name} podium: {top_3}")
        except Exception as e:
            print(f"⚠️ Could not fetch results for {event_name}: {e}")
    return past_race_results
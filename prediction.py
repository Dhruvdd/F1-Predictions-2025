import json
import pandas as pd
import fastf1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from data_preprocessing import get_race_data, load_race_adjustments

def train_model(selected_gp, year):
    """
    Trains a Gradient Boosting model using only current year's qualifying data.
    """
    # Load data: laps will be None in this mode, so we only get qualifying_data.
    _, qualifying_data = get_race_data(year, selected_gp)
    
    if qualifying_data is None or qualifying_data.empty:
        raise ValueError(f"🚨 Qualification data for {selected_gp} ({year}) is missing or empty!")
    
    merged_data = qualifying_data.copy()
    
    # Use only qualifying data; apply race adjustments only.
    race_adjustments = load_race_adjustments()
    merged_data["RaceAdjustment"] = merged_data["Driver"].map(race_adjustments).fillna(0).astype(int)
    merged_data["FinalPredictionTime"] = merged_data["QualifyingTime (s)"] + merged_data["RaceAdjustment"]
    
    # For feature set, we'll use only QualifyingTime (s)
    X = merged_data[["QualifyingTime (s)"]]
    y = merged_data["FinalPredictionTime"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    return model, merged_data

def calculate_podium_likelihood(previous_race_results):
    """Calculates the frequency (as a percentage) a driver finishes in the top 3 based on previous races."""
    podium_counts = {}
    total_races = len(previous_race_results)
    
    for race in previous_race_results:
        top3 = race[:3]
        for driver in top3:
            podium_counts[driver] = podium_counts.get(driver, 0) + 1
            
    podium_percentage = {driver: round((count / total_races) * 100, 2) for driver, count in podium_counts.items()}
    return podium_percentage

def fetch_last_year_gp_winner(year, gp_name):
    """Fetches the winner of last year's GP (this function may no longer be needed if we're removing previous year data)."""
    fastf1.Cache.enable_cache("f1_cache")
    events = fastf1.get_event_schedule(year - 1)
    event_row = events[events["EventName"].str.contains(gp_name, case=False, na=False)]
    if event_row.empty:
        print(f"🚨 GP '{gp_name}' not found for {year-1}.")
        return None
    event_round = int(event_row["RoundNumber"].values[0])
    try:
        session = fastf1.get_session(year - 1, event_round, "R")
        session.load(telemetry=False, weather=False)
        if session.results is not None and not session.results.empty:
            winner = session.results.sort_values("Position").iloc[0]["Abbreviation"]
            return winner
    except Exception as e:
        print(f"⚠️ Error fetching last year's winner for {gp_name}: {e}")
    return None

def fetch_past_race_results(year):
    """Fetches podium finishers for each race of the season (for podium probability calculation)."""
    fastf1.Cache.enable_cache("f1_cache")
    events = fastf1.get_event_schedule(year)
    results = []
    for idx, row in events.iterrows():
        event_round = int(row["RoundNumber"])
        event_name = row["EventName"]
        try:
            session = fastf1.get_session(year, event_round, "R")
            session.load(telemetry=False, weather=False)
            if session.results is not None and not session.results.empty:
                podium = session.results.sort_values("Position").head(3)["Abbreviation"].tolist()
                results.append(podium)
                print(f"✅ {event_name} podium: {podium}")
        except Exception as e:
            print(f"⚠️ Could not fetch results for {event_name}: {e}")
    return results

def predict_top3(model, qualifying_data, track_difficulty, previous_race_results, selected_gp, year):
    """
    Predicts the top 3 podium finishers and calculates podium likelihood based on previous race results.
    Returns:
        predicted_winner: predicted first place (string)
        podium: list of top 3 predicted drivers
        podium_probability: dictionary mapping drivers to their podium probability (%)
    """
    # Predict race times using model
    X = qualifying_data[["QualifyingTime (s)"]]
    qualifying_data["PredictedTime"] = model.predict(X)
    
    # Sort drivers by predicted time (lower is better)
    sorted_df = qualifying_data.sort_values(by="PredictedTime", ascending=True)
    
    podium = []
    for driver in sorted_df["Driver"]:
        if driver not in podium:
            podium.append(driver)
        if len(podium) == 3:
            break
    predicted_winner = podium[0]
    
    # Calculate podium probability based on previous race results (if available)
    podium_probability = {}
    if previous_race_results:
        podium_percentage = calculate_podium_likelihood(previous_race_results)
        podium_probability = podium_percentage
    
    return predicted_winner, podium, podium_probability
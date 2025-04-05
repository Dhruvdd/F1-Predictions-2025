import streamlit as st
import pandas as pd
import fastf1
from data_preprocessing import get_grand_prix_list
from prediction import train_model, predict_top3

fastf1.Cache.enable_cache("f1_cache")

st.title("🏎️ F1 Podium Predictor")

year = st.selectbox("Select Season Year:", [2025, 2024])
gp_list = get_grand_prix_list(year)
selected_gp = st.selectbox("Select a Grand Prix:", gp_list)

st.info("Using current year qualifying data only for predictions.")

# Here's the button to predict podium finishers
if st.button("Predict Podium Finishers"):
    try:
        model, merged_data = train_model(selected_gp, year)  # Notice no use_past_race_data param
        # You might skip previous race results if you no longer need them
        predicted_winner, podium, podium_probability = predict_top3(
            model, merged_data, track_difficulty=1.3,
            previous_race_results=None,  # if not needed
            selected_gp=selected_gp, year=year
        )

        st.success(f"🏆 Predicted Winner: {predicted_winner}")
        st.subheader("🥇 Predicted Podium")
        st.write(f"🥇 1st: {podium[0]}")
        st.write(f"🥈 2nd: {podium[1]}")
        st.write(f"🥉 3rd: {podium[2]}")
    except Exception as e:
        st.error(f"Error: {e}")
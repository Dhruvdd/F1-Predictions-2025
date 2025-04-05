# F1 Podium Predictor

## Overview

The F1 Podium Predictor is a dynamic application that leverages historical Formula 1 data (via the FastF1 library) and current season qualifying information to predict the podium finishers for a selected Grand Prix. The project uses a Gradient Boosting Regressor from scikit-learn to estimate race performance based solely on current qualifying data. It also calculates podium likelihood based on past race results, and it integrates various adjustment factors (race adjustments and driver consistency) stored in JSON files.

### Features

Dynamic Data Loading:
- Retrieves season schedules and session data (qualifying sessions) dynamically from the FastF1 API.

Driver Mapping:
- Uses a JSON file (driver_mapping.json) to map FastF1 driver codes (e.g., VER) to full driver names (e.g., “Max Verstappen”). This ensures consistency and ease of update.

Race Adjustments & Driver Consistency:
- Applies adjustment factors from race_adjustments.json and driver_consistency.json to better reflect historical performance trends.

Model Training:
- Trains a multi-feature Gradient Boosting model (using qualifying times and, if available, sector times) to predict lap performance.

Podium Prediction:
- Predicts the winner and the top 3 podium finishers using current season qualifying data.
- Optionally, it can calculate the podium likelihood based on previous race results.

Streamlit Interface:
- Provides an interactive web app interface where users can select the season, Grand Prix, and view predictions along with additional statistics.


## Installation

### Prerequisites
Python 3.8+ is recommended.

Install required Python packages using pip: 

	• pip install fastf1 pandas numpy scikit-learn streamlit

## Usage
Run the App:

Start the Streamlit web application by running:

	• streamlit run app.py

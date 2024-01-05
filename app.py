# app.py

import datetime
import streamlit as st
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv('matches.csv')
print(df.info())
# Load the model
with open('score_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def calculate_crr(current_score, overs):
    return round((current_score * 6)/ overs, 2)

def main():
    st.title("Cricket Score Prediction")

    # Create input widgets
    st.sidebar.header("User Input")
    season = st.sidebar.slider("Season", 2008, 2050, 2023)
    team1 = st.sidebar.selectbox("Team1", df['team1'].unique())
    team2 = st.sidebar.selectbox("Team2", df['team2'].unique())
    city = st.sidebar.selectbox("City", df['city'].unique())
    venue = st.sidebar.selectbox("Venue", df['venue'].unique())
    toss_winner = st.sidebar.selectbox("Toss Won By", [team1, team2])
    toss_decision = st.sidebar.selectbox("Decision", df['toss_decision'].unique())
    if toss_decision == 'bat':
        current_innings = toss_winner
        other_team = team2 if toss_winner == team1 else team1
    else:
        current_innings = team2 if toss_winner == team1 else team2
        other_team = toss_winner
    current_innings = st.sidebar.selectbox("batting", [team1, team2])
    other_team = st.sidebar.selectbox("bowling", [team1, team2])
    balls = st.sidebar.number_input("Balls", min_value=30, max_value=120)
    current_score = st.sidebar.number_input("Enter the current score", min_value=0, max_value=300)
    last5_runs = st.sidebar.number_input("Enter last5 overs runs", min_value=0, max_value=current_score)
    wickets = st.sidebar.slider("Wickets", 0, 10, 4)
    month = st.sidebar.slider("Month", 3, 6, 4)
    day = st.sidebar.slider("Day", 1, 31, 11)
    date_str = f"{season}-{month}-{day}"
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    weekday = date_obj.weekday()

    wickets_left = 10 - wickets
    crr = round(current_score*6/balls, 2)
    balls_left = (120-balls)

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'id': [df['id'].max()+1],  # The value doesn't matter, it's just a placeholder
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'city': [city],
        'venue': [venue],
        'season': [season],
        'team1': [team1],
        'team2': [team2],
        'batting_team': [current_innings],
        'bowling_team': [other_team],
#        'innings_id': [innings_id],
        'current_score': [current_score],
        'wickets': [wickets],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'balls_left': balls_left,
        'last5_runs': [last5_runs],
        'month': [month],
        'day': [day],
        'weekday': [weekday]
    })
    print(user_input.to_string())
    user_input['avg_runs_city'] = df.groupby('city')['total'].transform('median')
    user_input['avg_runs_city'] = df['avg_runs_city'].ffill()

#    user_input['critical'] = (user_input['wickets_left'] == 1)
#    user_input['ratio'] = (user_input['wickets_left'].div(user_input['balls_left'])).replace([np.inf, -np.inf, np.nan], 0)

    # Display user input
    st.subheader("User Input:")
#    st.write(user_input)

    # Predict button
    if st.button("Predict"):
        # Make predictions
        prediction = model.predict(user_input).astype(int)
        st.sidebar.header("Model Prediction")
        st.sidebar.write("Predicted Score:", prediction[0])

if __name__ == "__main__":
    main()



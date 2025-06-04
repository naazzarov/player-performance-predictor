from joblib import load
import pandas as pd
import numpy as np

# Load your trained model
model = load('player_performance_predictor.joblib')  # Make sure the file path is correct

# Function that predicts player's stats once it's called and takes player's name as argument
def predict_player_future(player_name, dataset, seasons=3, current_age=30):
    player_data = dataset[dataset['Players'] == player_name]
    if player_data.empty:
        raise ValueError(f"Player '{player_name}' not found.")
    
    latest_season = player_data.sort_values('Seasons').iloc[-1].copy()
    
    # Extract base year (handles '2024' or '2023/2024')
    if '/' in str(latest_season['Seasons']):
        latest_year = int(str(latest_season['Seasons']).split('/')[-1])
    else:
        latest_year = int(latest_season['Seasons'])
    
    predictions = []
    current_goals = latest_season['Goals']
    current_assists = latest_season['Assists']
    current_matches = latest_season['Matches']
    
    for year in range(1, seasons + 1):
        # --- Simulate Realistic Variance ---
        # 1. Random fluctuation (±10% stats)
        goals_change = np.random.uniform(0.9, 1.1)  # Random growth/decline
        assists_change = np.random.uniform(0.9, 1.1)
        matches_change = np.random.uniform(0.95, 1.05)
        
        # Update stats
        current_matches = int(current_matches * matches_change)
        current_goals = int(current_goals * goals_change)
        current_assists = int(current_assists * assists_change)
        
        # Ensure stats don't drop below 0
        current_matches = max(current_matches, 5)
        current_goals = max(current_goals, 0)
        current_assists = max(current_assists, 0)
        
        # --- Calculate Features ---
        goals_per_match = current_goals / current_matches if current_matches > 0 else 0
        assists_per_match = current_assists / current_matches if current_matches > 0 else 0
        contributions_per_match = (current_goals + current_assists) / current_matches if current_matches > 0 else 0
        
        # --- Predict Rating ---
        features = pd.DataFrame([{
            'Matches': current_matches,
            'Goals': current_goals,
            'Assists': current_assists,
            'Goals_per_Match': goals_per_match,
            'Assists_per_Match': assists_per_match,
            'Contributions_per_Match': contributions_per_match
        }])
        
        projected_rating = model.predict(features)[0]
        
        predictions.append({
            'Player': player_name,
            'Season': f"{latest_year + year}",
            'Predicted Rating': round(projected_rating, 2),
            'Goals': current_goals,
            'Assists': current_assists,
            'Matches': current_matches,
        })
    
    return pd.DataFrame(predictions)

# Load your dataset
df = pd.read_csv('players.csv')

# Predict for a player (e.g., "Lionel Messi")
player = input("Player To Predict: ")
future_performance = predict_player_future(player, df, seasons=3)
print(future_performance)
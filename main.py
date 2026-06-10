from joblib import load
import pandas as pd
import numpy as np

model = load('player_performance_predictor.joblib')

TEAMS = sorted([
    "AC Milan", "AL Hilal", "AL Ittihad", "AL Nassr", "Ajax", "Alaves",
    "Arsenal", "Aston Villa", "Atalanta", "Atletico Madrid",
    "Augsburg", "Barcelona", "Bayer Leverkusen", "Bayern Munich",
    "Betis", "Bologna", "Bournemouth", "Brentford", "Brighton",
    "Brugge", "Burnley", "Cagliari", "Celtic", "Chelsea",
    "Club Brugge", "Crystal Palace", "Dortmund", "Everton",
    "Fiorentina", "Frankfurt", "Freiburg", "Fulham", "Genoa",
    "Getafe", "Girona", "Gladbach", "Granada", "Heidenheim",
    "Hoffenheim", "InterMiami", "Inter Milan", "Juventus",
    "Lazio", "Leeds", "Lens", "Leverkusen", "Lille", "Liverpool",
    "Lyon", "Mainz", "Mallorca", "Manchester City", "Manchester Utd",
    "Marseille", "Milan", "Monaco", "Monza", "Napoli",
    "Newcastle Utd", "Nice", "Nottingham Forest", "OL",
    "OL Marseille", "Osasuna", "PSG", "PSV", "Palmeiras",
    "Parma", "Rayo Vallecano", "Real Betis", "Real Madrid",
    "Real Sociedad", "Roma", "Rangers", "RB Leipzig", "Reims",
    "Rennes", "Salernitana", "Sassuolo", "Schalke", "Sevilla",
    "Sheffield Utd", "Southampton", "Spezia", "Strasbourg",
    "Stuttgart", "Sunderland", "Torino", "Tottenham", "Toulouse",
    "Udinese", "Union Berlin", "Valencia", "Valladolid",
    "Venice", "Verona", "Villarreal", "Watford", "West Ham",
    "Wolfsburg", "Wolves", "Zenit", "real Madrid"
])

def get_team_columns(team_name):
    prefix = "Teams_"
    row = {}
    for t in TEAMS:
        row[f"{prefix}{t}"] = team_name == t
    return row

def predict_player_future(player_name, dataset, seasons=3, current_age=30):
    player_data = dataset[dataset['Players'] == player_name]
    if player_data.empty:
        raise ValueError(f"Player '{player_name}' not found.")

    latest_season = player_data.sort_values('Seasons').iloc[-1].copy()

    if '/' in str(latest_season['Seasons']):
        latest_year = int(str(latest_season['Seasons']).split('/')[-1])
    else:
        latest_year = int(latest_season['Seasons'])

    player_team = latest_season['Teams']
    team_cols = get_team_columns(player_team)

    predictions = []
    current_goals = int(latest_season['Goals'])
    current_assists = int(latest_season['Assists'])
    current_matches = int(latest_season['Matches'])

    for year in range(1, seasons + 1):
        age = current_age + year

        goals_change = np.random.uniform(0.9, 1.1)
        assists_change = np.random.uniform(0.9, 1.1)
        matches_change = np.random.uniform(0.95, 1.05)

        if age > 30:
            penalty = 1.0 - (age - 30) * 0.03
            goals_change *= penalty
            assists_change *= penalty

        current_matches = max(int(current_matches * matches_change), 5)
        current_goals = max(int(current_goals * goals_change), 0)
        current_assists = max(int(current_assists * assists_change), 0)

        goals_per_match = current_goals / current_matches if current_matches > 0 else 0
        assists_per_match = current_assists / current_matches if current_matches > 0 else 0
        contributions_per_match = (current_goals + current_assists) / current_matches if current_matches > 0 else 0

        features = pd.DataFrame([{
            'Matches': current_matches,
            'Goals': current_goals,
            'Assists': current_assists,
            'Goals_per_Match': goals_per_match,
            'Assists_per_Match': assists_per_match,
            'Contributions_per_Match': contributions_per_match,
            **team_cols
        }])

        projected_rating = model.predict(features)[0]

        predictions.append({
            'Player': player_name,
            'Season': f"{latest_year + year}",
            'Age': age,
            'Predicted Rating': round(projected_rating, 2),
            'Goals': current_goals,
            'Assists': current_assists,
            'Matches': current_matches,
        })

    return pd.DataFrame(predictions)

df = pd.read_csv('players.csv')

if __name__ == "__main__":
    try:
        player = input("Enter player name to predict: ").strip()
        if not player:
            print("Error: Please enter a valid player name")
        else:
            future_performance = predict_player_future(player, df, seasons=3)
            print("\nPredictions for the next 3 seasons:")
            print(future_performance.to_string(index=False))
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: players.csv or model file not found")

# ⚽ Player Performance Predictor

A machine learning project that predicts the future performance ratings and statistics of football (soccer) players based on their historical data.

## 📖 Project Overview

This project uses a trained machine learning model to forecast how professional football players will perform in the next 3 seasons. It analyzes player statistics like goals, assists, and matches played to estimate their future ratings.

**Key Features:**
- Predicts player ratings for the next 3 seasons
- Estimates future goals, assists, and matches played
- Includes realistic variance and fluctuations in predictions
- Works with any player in the dataset

## 📊 Dataset

The dataset contains historical player performance data including:
- **Teams**: The club the player plays for
- **Seasons**: Season year(s) (e.g., 2024, 2023/2024)
- **Players**: Player name
- **Matches**: Number of matches played
- **Goals**: Number of goals scored
- **Assists**: Number of assists made
- **Seasons Ratings**: Player rating/score for that season

Example data:
```
InterMiami, 2024, Lionel Messi, 15 matches, 14 goals, 11 assists, 8.5 rating
```

## 🚀 How to Use

### 1. **Load and Clean Data**
```bash
python load_data.py
```
This script:
- Loads the `players.csv` file
- Removes rows with missing ratings
- Fills missing numeric values with the median
- Shows data preview

### 2. **Make Predictions**
```bash
python main.py
```
When prompted, enter a player's name:
```
Player To Predict: Lionel Messi
```

The program will output predictions for the next 3 seasons:
```
         Player  Season  Predicted Rating  Goals  Assists  Matches
0  Lionel Messi    2025              8.3     13        10       14
1  Lionel Messi    2026              8.1     12         9       13
2  Lionel Messi    2027              7.9     11         8       12
```

## 🤖 How It Works

1. **Model Loading**: The pre-trained model (`player_performance_predictor.joblib`) is loaded
2. **Player Search**: Finds the latest season data for the requested player
3. **Feature Calculation**: Computes key metrics:
   - Goals per match
   - Assists per match
   - Total contributions per match
4. **Predictions**: For each future season:
   - Applies realistic variance (±10% fluctuation)
   - Calculates derived features
   - Uses the ML model to predict rating
   - Returns all predicted statistics

## 📁 Project Structure

```
player-performance-predictor/
├── README.md                              # This file
├── main.py                               # Main prediction script
├── load_data.py                          # Data loading and cleaning
├── players.csv                           # Player statistics dataset
├── player_performance_predictor.joblib   # Trained ML model
├── player_rating_model.pkl               # Alternative model file
├── main.ipynb                            # Jupyter notebook for main logic
├── load_data.ipynb                       # Jupyter notebook for data loading
└── model_training.ipynb                  # Jupyter notebook for model training
```

## 🛠️ Requirements

- Python 3.x
- pandas
- numpy
- joblib
- scikit-learn (for model)

Install with:
```bash
pip install pandas numpy joblib scikit-learn
```

## 📈 Example Predictions

The model predicts realistic performance changes considering:
- Player age and career trajectory
- Historical performance trends
- Random fluctuations (like good/bad seasons)
- Minimum thresholds (to prevent unrealistic negative stats)

## 💡 Notes

- Enter the exact player name as it appears in the dataset
- Predictions are probabilistic and include realistic variance
- The model works best for players with multiple seasons of data
- Future predictions assume similar playing conditions

## 👨‍💻 Author

Student project for predicting and analyzing football player performance trends.

---

**Enjoy predicting! ⚽**

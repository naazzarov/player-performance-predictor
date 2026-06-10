import pandas as pd

df = pd.read_csv("players.csv")

df.dropna(subset=["Seasons Ratings"], inplace=True)

feature_cols = ["Matches", "Goals", "Assists"]
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFirst few rows of the dataset:")
print(df.head())

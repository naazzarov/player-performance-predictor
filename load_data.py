import pandas as pd

# Load the dataset
df = pd.read_csv("players.csv")

# Drop rows with missing ratings
df = df.dropna(subset=["Seasons Ratings"])

# Only fill missing numeric columns with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Show any remaining missing values
print("\n❓ Missing values after cleaning:")
print(df.isnull().sum())

# Show a preview of the cleaned data
print("\n📋 First few rows of the dataset:")
print(df.head())

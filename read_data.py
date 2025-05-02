import pandas as pd

# Read the CSV file
df = pd.read_csv("application_train.csv")

# Dealing with missing values
# Numeric - replace with mean
df = df.fillna(df.mean(numeric_only=True))

# Categorical - replace with the mode
df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col)

# Check if no Na's
remaining_missing = df.isna().sum().sum()
print(f"Remaining missing values: {remaining_missing}")


df.to_csv('cleaned_application_train.csv', index=False)
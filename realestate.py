import pandas as pd
import numpy as np

# Load dataset (update path)
file_path = "maryland_real_property.csv"
df = pd.read_csv(file_path, low_memory=False)

print("Dataset Loaded")
print(f"Shape: {df.shape}")
print("\n")

#basic info
print("BASIC INFO")
print(df.info())
print("\n")

#variable types
categorical_cols = []
discrete_cols = []
continuous_cols = []

for col in df.columns:
    if df[col].dtype == "object":
        categorical_cols.append(col)
    else:
        unique_vals = df[col].nunique()
        
        # Heuristic rules
        if unique_vals < 50:
            categorical_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

print("VARIABLE TYPES")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols[:10]} ...")
print(f"Discrete ({len(discrete_cols)}): {discrete_cols[:10]} ...")
print(f"Continuous ({len(continuous_cols)}): {continuous_cols[:10]} ...")
print("\n")

#missing values
print("MISSING VALUES")
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print(missing.head(20))
print("\n")

#summary
print("NUMERICAL SUMMARY")
print(df.describe())
print("\n")

#categorical analysis
print("TOP CATEGORIES")
for col in categorical_cols[:5]: 
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))

#discrete analysis
print("\nDISCRETE VARIABLE DISTRIBUTIONS")
for col in discrete_cols[:5]:  # limit output
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))


print("\nCORRELATION MATRIX (Top)")
corr = df[continuous_cols].corr()
#strongest correlations
corr_pairs = (
    corr.abs()
    .unstack()
    .sort_values(ascending=False)
    .drop_duplicates()
)

print(corr_pairs[1:15])  # skip self-correlation

report = {
    "categorical": categorical_cols,
    "discrete": discrete_cols,
    "continuous": continuous_cols
}

pd.Series(report).to_json("variable_types.json")

print("\nAnalysis Complete. Results saved.")

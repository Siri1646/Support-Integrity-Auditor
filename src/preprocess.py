import pandas as pd

# Load dataset
df = pd.read_csv("data/customer_support_tickets.csv")

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nPriority Distribution")
print(df["Priority_Level"].value_counts())

print("\nMissing Values")
print(df.isnull().sum())

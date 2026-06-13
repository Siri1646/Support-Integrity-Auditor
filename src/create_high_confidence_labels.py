import pandas as pd

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

# Keep only confident examples:
# very low or very high severity scores, or strong severity delta
high_conf = df[
    (df["Severity_Score"] <= 25) |
    (df["Severity_Score"] >= 70) |
    (df["Severity_Delta"].abs() >= 2)
].copy()

print("Original rows:", len(df))
print("High-confidence rows:", len(high_conf))

print("\nLabel distribution:")
print(high_conf["Mismatch_Label"].value_counts())

high_conf.to_csv("data/high_confidence_tickets.csv", index=False)
print("\nSaved: data/high_confidence_tickets.csv")
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

def label_from_score(score):
    if score < 30:
        return "Low"
    elif score < 55:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"

df["Keyword_Severity"] = df["Text_Urgency_Score"].apply(label_from_score)
df["Resolution_Severity"] = df["Resolution_Score"].apply(label_from_score)
df["Category_Severity"] = df["Category_Score"].apply(label_from_score)

df["Keyword_Mismatch"] = (df["Keyword_Severity"] != df["Priority_Level"]).astype(int)
df["Resolution_Mismatch"] = (df["Resolution_Severity"] != df["Priority_Level"]).astype(int)
df["Category_Mismatch"] = (df["Category_Severity"] != df["Priority_Level"]).astype(int)

def agreement(a, b):
    return round((df[a] == df[b]).mean() * 100, 2)

print("Pseudo-Label Signal Agreement")
print("--------------------------------")
print("Keyword vs Resolution:", agreement("Keyword_Mismatch", "Resolution_Mismatch"), "%")
print("Keyword vs Category:", agreement("Keyword_Mismatch", "Category_Mismatch"), "%")
print("Resolution vs Category:", agreement("Resolution_Mismatch", "Category_Mismatch"), "%")

print("\nAblation Study Against Final Pseudo Label")
print("--------------------------------")

signals = {
    "Keyword only": "Keyword_Mismatch",
    "Resolution only": "Resolution_Mismatch",
    "Category only": "Category_Mismatch"
}

for name, col in signals.items():
    acc = accuracy_score(df["Mismatch_Label"], df[col])
    f1 = f1_score(df["Mismatch_Label"], df[col], average="macro")
    print(f"{name}: Accuracy={acc:.3f}, Macro F1={f1:.3f}")

print("\nFinal Fusion:")
print("Accuracy=1.000, Macro F1=1.000")
print("Note: Final fusion is the pseudo-label target generated using weighted signal fusion.")
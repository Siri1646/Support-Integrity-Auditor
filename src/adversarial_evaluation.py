import pandas as pd
import joblib

MODEL_PATH = "models/sia_baseline_model.pkl"
DATA_PATH = "data/adversarial_tickets.csv"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

df["Combined_Text"] = (
    df["Ticket_Subject"].fillna("") + " " +
    df["Ticket_Description"].fillna("")
)

X = df[[
    "Combined_Text",
    "Issue_Category",
    "Ticket_Channel",
    "Resolution_Time_Hours",
    "Satisfaction_Score"
]]

predictions = model.predict(X)

df["Predicted_Label"] = predictions
df["Predicted_Class"] = df["Predicted_Label"].map({
    0: "Consistent",
    1: "Mismatch"
})

# Expected labels are derived from adversarial intent:
# odd/easy false alarm examples and hidden crisis examples were manually designed.
expected = [
    1,  # Password reset request marked Critical -> False Alarm mismatch
    1,  # Dashboard unavailable marked Low -> Hidden Crisis mismatch
    1,  # Payment inquiry marked High -> False Alarm mismatch
    1,  # Fraud suspected marked Medium -> Hidden Crisis mismatch
    1,  # Feature request marked Critical -> False Alarm mismatch
    1,  # Account locked marked Low -> Hidden Crisis mismatch
    1,  # Login issue resolved marked High -> False Alarm mismatch
    1,  # Data sync failure marked Low -> Hidden Crisis mismatch
    1,  # Billing clarification marked Critical -> False Alarm mismatch
    1   # Security breach marked Medium -> Hidden Crisis mismatch
]

df["Expected_Label"] = expected
df["Correct"] = df["Predicted_Label"] == df["Expected_Label"]

score = int(df["Correct"].sum())
total = len(df)

print("Adversarial Evaluation")
print("----------------------")
print(f"Tickets Tested: {total}")
print(f"Correct Predictions: {score}")
print(f"Robustness Score: {round((score / total) * 100, 2)}%")

print("\nDetailed Results:")
print(df[[
    "Ticket_Subject",
    "Priority_Level",
    "Predicted_Class",
    "Correct"
]].to_string(index=False))

df.to_csv("data/adversarial_results.csv", index=False)
print("\nSaved: data/adversarial_results.csv")
import pandas as pd
import re

df = pd.read_csv("data/customer_support_tickets.csv")

category_scores = {
    "General Inquiry": 20,
    "Account": 50,
    "Billing": 60,
    "Technical": 75,
    "Fraud": 90
}

priority_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

reverse_priority_map = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical"
}

urgent_keywords = {
    "failed": 20,
    "failing": 20,
    "cannot": 20,
    "can't": 20,
    "unable": 20,
    "not loading": 25,
    "crashes": 25,
    "crash": 25,
    "password": 15,
    "login": 20,
    "payment": 20,
    "refund": 15,
    "fraud": 35,
    "locked": 25,
    "error": 20,
    "data not syncing": 25,
    "subscription": 10,
    "2fa": 20
}

def resolution_score(hours):
    if hours <= 12:
        return 20
    elif hours <= 24:
        return 40
    elif hours <= 48:
        return 60
    elif hours <= 72:
        return 80
    else:
        return 100

def text_urgency_score(subject, description):
    text = f"{subject} {description}".lower()
    score = 0
    matched = []

    for keyword, weight in urgent_keywords.items():
        if keyword in text:
            score += weight
            matched.append(keyword)

    return min(score, 100), ", ".join(matched)

def infer_severity(score):
    if score < 30:
        return "Low"
    elif score < 55:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"

def mismatch_type(row):
    assigned = priority_map[row["Priority_Level"]]
    inferred = priority_map[row["Inferred_Severity"]]

    if inferred > assigned:
        return "Hidden Crisis"
    elif inferred < assigned:
        return "False Alarm"
    else:
        return "Consistent"

df["Category_Score"] = df["Issue_Category"].map(category_scores)
df["Resolution_Score"] = df["Resolution_Time_Hours"].apply(resolution_score)

text_results = df.apply(
    lambda row: text_urgency_score(
        row["Ticket_Subject"],
        row["Ticket_Description"]
    ),
    axis=1
)

df["Text_Urgency_Score"] = text_results.apply(lambda x: x[0])
df["Matched_Keywords"] = text_results.apply(lambda x: x[1])

df["Severity_Score"] = (
    0.35 * df["Category_Score"] +
    0.35 * df["Resolution_Score"] +
    0.30 * df["Text_Urgency_Score"]
)

df["Inferred_Severity"] = df["Severity_Score"].apply(infer_severity)

df["Mismatch_Label"] = (
    df["Priority_Level"] != df["Inferred_Severity"]
).astype(int)

df["Mismatch_Type"] = df.apply(mismatch_type, axis=1)

df["Severity_Delta"] = df.apply(
    lambda row: priority_map[row["Inferred_Severity"]] - priority_map[row["Priority_Level"]],
    axis=1
)

df.to_csv("data/pseudo_labeled_tickets.csv", index=False)

print("Pseudo-labeling completed.")
print("Saved file: data/pseudo_labeled_tickets.csv")

print("\nMismatch Label Distribution:")
print(df["Mismatch_Label"].value_counts())

print("\nMismatch Type Distribution:")
print(df["Mismatch_Type"].value_counts())

print("\nSample Output:")
print(
    df[
        [
            "Ticket_ID",
            "Ticket_Subject",
            "Priority_Level",
            "Inferred_Severity",
            "Severity_Score",
            "Matched_Keywords",
            "Mismatch_Type"
        ]
    ].head(10).to_string(index=False)
)
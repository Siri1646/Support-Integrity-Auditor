import pandas as pd
import joblib
import json

model = joblib.load("models/sia_baseline_model.pkl")

priority_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

urgent_keywords = [
    "failed", "failing", "cannot", "unable", "not loading",
    "crashes", "crash", "password", "login", "payment",
    "refund", "fraud", "locked", "error", "data not syncing",
    "subscription", "2fa"
]

def detect_keywords(subject, description):
    text = f"{subject} {description}".lower()
    return [word for word in urgent_keywords if word in text]

def get_mismatch_type(ticket):
    assigned = priority_map[ticket["Priority_Level"]]

    # Simple severity estimate for explanation
    severity_score = 0

    if ticket["Issue_Category"] == "Fraud":
        severity_score += 3
    elif ticket["Issue_Category"] in ["Technical", "Billing"]:
        severity_score += 2
    else:
        severity_score += 1

    if ticket["Resolution_Time_Hours"] > 72:
        severity_score += 3
    elif ticket["Resolution_Time_Hours"] > 48:
        severity_score += 2
    else:
        severity_score += 1

    if ticket["Satisfaction_Score"] <= 2:
        severity_score += 2

    inferred = min(4, max(1, round(severity_score / 2)))

    if inferred > assigned:
        return "Hidden Crisis", inferred - assigned
    elif inferred < assigned:
        return "False Alarm", inferred - assigned
    else:
        return "Consistent", 0

def predict_ticket(ticket):
    df = pd.DataFrame([ticket])

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

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]

    matched_keywords = detect_keywords(
        ticket["Ticket_Subject"],
        ticket["Ticket_Description"]
    )

    mismatch_type, severity_delta = get_mismatch_type(ticket)

    dossier = {
        "ticket_id": ticket["Ticket_ID"],
        "assigned_priority": ticket["Priority_Level"],
        "prediction": "Mismatch" if prediction == 1 else "Consistent",
        "mismatch_type": mismatch_type if prediction == 1 else "Consistent",
        "severity_delta": severity_delta,
        "feature_evidence": [
            {
                "signal": "keyword",
                "value": ", ".join(matched_keywords) if matched_keywords else "No strong urgency keyword detected",
                "weight": "high" if matched_keywords else "low"
            },
            {
                "signal": "resolution_time",
                "value": f'{ticket["Resolution_Time_Hours"]} hours',
                "interpretation": "Long resolution time suggests higher severity"
                if ticket["Resolution_Time_Hours"] > 48
                else "Resolution time does not indicate severe delay"
            },
            {
                "signal": "issue_category",
                "value": ticket["Issue_Category"],
                "interpretation": "Issue category used as structured severity signal"
            }
        ],
        "constraint_analysis": (
            f'The ticket was assigned {ticket["Priority_Level"]}. '
            f'The model found evidence from ticket text, issue category, '
            f'resolution time, and satisfaction score to judge whether the priority is mismatched.'
        ),
        "confidence": round(float(confidence), 3)
    }

    return dossier

sample_ticket = {
    "Ticket_ID": "TEST-001",
    "Ticket_Subject": "Login failed urgently",
    "Ticket_Description": "Customer cannot login even after password reset. Dashboard is not loading.",
    "Issue_Category": "Technical",
    "Priority_Level": "Low",
    "Ticket_Channel": "Chat",
    "Resolution_Time_Hours": 65,
    "Satisfaction_Score": 1
}

output = predict_ticket(sample_ticket)

print(json.dumps(output, indent=4))
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = "models/sia_baseline_model.pkl"


URGENT_KEYWORDS = [
    "failed", "failing", "cannot", "can't", "unable", "not loading",
    "crashes", "crash", "password", "login", "payment", "refund",
    "fraud", "locked", "error", "data not syncing", "subscription", "2fa"
]

PRIORITY_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

REVERSE_PRIORITY_MAP = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical"
}


def detect_keywords(subject, description):
    text = f"{subject} {description}".lower()
    return [word for word in URGENT_KEYWORDS if word in text]


def estimate_inferred_severity(row):
    score = 0

    category = row["Issue_Category"]
    hours = row["Resolution_Time_Hours"]
    satisfaction = row["Satisfaction_Score"]

    if category == "Fraud":
        score += 4
    elif category == "Technical":
        score += 3
    elif category == "Billing":
        score += 2
    else:
        score += 1

    if hours > 72:
        score += 4
    elif hours > 48:
        score += 3
    elif hours > 24:
        score += 2
    else:
        score += 1

    if satisfaction <= 2:
        score += 2
    elif satisfaction == 3:
        score += 1

    severity_num = min(4, max(1, round(score / 2)))
    return REVERSE_PRIORITY_MAP[severity_num]


def create_dossier(row, prediction, confidence):
    matched_keywords = detect_keywords(
        row["Ticket_Subject"],
        row["Ticket_Description"]
    )

    inferred_severity = estimate_inferred_severity(row)
    assigned_priority = row["Priority_Level"]

    severity_delta = (
        PRIORITY_MAP[inferred_severity] -
        PRIORITY_MAP[assigned_priority]
    )

    if prediction == 0:
        mismatch_type = "Consistent"
    elif severity_delta > 0:
        mismatch_type = "Hidden Crisis"
    elif severity_delta < 0:
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Mismatch"

    dossier = {
        "ticket_id": row["Ticket_ID"],
        "assigned_priority": assigned_priority,
        "inferred_severity": inferred_severity,
        "mismatch_type": mismatch_type,
        "severity_delta": int(severity_delta),
        "feature_evidence": [
            {
                "signal": "keyword",
                "value": ", ".join(matched_keywords) if matched_keywords else "No strong urgency keyword detected",
                "weight": "high" if matched_keywords else "low"
            },
            {
                "signal": "resolution_time",
                "value": f'{row["Resolution_Time_Hours"]} hours',
                "interpretation": (
                    "Long resolution time suggests higher severity"
                    if row["Resolution_Time_Hours"] > 48
                    else "Resolution time does not indicate severe delay"
                )
            },
            {
                "signal": "issue_category",
                "value": row["Issue_Category"],
                "interpretation": "Issue category used as structured severity signal"
            },
            {
                "signal": "ticket_channel",
                "value": row["Ticket_Channel"],
                "interpretation": "Channel used as structured metadata feature"
            }
        ],
        "constraint_analysis": (
            f'This dossier only uses fields present in the input ticket: '
            f'text, issue category, channel, resolution time, and satisfaction score. '
            f'The assigned priority was {assigned_priority}, while the inferred severity was {inferred_severity}.'
        ),
        "confidence": round(float(confidence), 3)
    }

    return dossier


def predict_dataframe(input_df):
    model = joblib.load(MODEL_PATH)

    df = input_df.copy()

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
    probabilities = model.predict_proba(X)

    df["Prediction_Label"] = predictions
    df["Prediction"] = [
        "Mismatch" if pred == 1 else "Consistent"
        for pred in predictions
    ]

    df["Confidence"] = [
        round(float(probabilities[i][pred]), 3)
        for i, pred in enumerate(predictions)
    ]

    dossiers = []
    for i, row in df.iterrows():
        dossier = create_dossier(
            row,
            df.loc[i, "Prediction_Label"],
            df.loc[i, "Confidence"]
        )
        dossiers.append(json.dumps(dossier))

    df["Evidence_Dossier"] = dossiers

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Support Integrity Auditor batch prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        default="data/predictions_output.csv",
        help="Path to save output CSV file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    result_df = predict_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("Batch prediction completed.")
    print(f"Input: {input_path}")
    print(f"Output saved to: {output_path}")
    print("\nPrediction Distribution:")
    print(result_df["Prediction"].value_counts())


if __name__ == "__main__":
    main()
    
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("models/sia_baseline_model.pkl")

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")

st.title("Support Integrity Auditor")
st.caption("Self-supervised CRM ticket priority mismatch detection")

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

# KPI cards
total_tickets = len(df)
mismatch_count = int(df["Mismatch_Label"].sum())
mismatch_rate = round((mismatch_count / total_tickets) * 100, 2)
hidden_crisis = int((df["Mismatch_Type"] == "Hidden Crisis").sum())
false_alarm = int((df["Mismatch_Type"] == "False Alarm").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tickets", total_tickets)
col2.metric("Mismatch Rate", f"{mismatch_rate}%")
col3.metric("Hidden Crises", hidden_crisis)
col4.metric("False Alarms", false_alarm)

st.divider()

# Sidebar
st.sidebar.header("Audit a New Ticket")

ticket_id = st.sidebar.text_input("Ticket ID", "TEST-001")
subject = st.sidebar.text_input("Ticket Subject", "Login failed urgently")
description = st.sidebar.text_area(
    "Ticket Description",
    "Customer cannot login even after password reset. Dashboard is not loading."
)
issue_category = st.sidebar.selectbox(
    "Issue Category",
    ["Technical", "Billing", "Account", "General Inquiry", "Fraud"]
)
priority = st.sidebar.selectbox(
    "Assigned Priority",
    ["Low", "Medium", "High", "Critical"]
)
channel = st.sidebar.selectbox(
    "Ticket Channel",
    ["Chat", "Email", "Web Form"]
)
resolution_time = st.sidebar.number_input("Resolution Time Hours", min_value=0, value=65)
satisfaction = st.sidebar.slider("Satisfaction Score", min_value=1, max_value=5, value=1)

st.sidebar.divider()
st.sidebar.subheader("Batch CSV Audit")
uploaded_file = st.sidebar.file_uploader("Upload ticket CSV", type=["csv"])


def predict(ticket_df):
    temp = ticket_df.copy()
    temp["Combined_Text"] = (
        temp["Ticket_Subject"].fillna("") + " " +
        temp["Ticket_Description"].fillna("")
    )

    X = temp[[
        "Combined_Text",
        "Issue_Category",
        "Ticket_Channel",
        "Resolution_Time_Hours",
        "Satisfaction_Score"
    ]]

    preds = model.predict(X)
    probs = model.predict_proba(X)

    return preds, probs


def create_dossier(ticket, pred, confidence):
    prediction_text = "Mismatch" if pred == 1 else "Consistent"

    matched_keywords = []
    keyword_bank = [
        "failed", "failing", "cannot", "unable", "not loading", "crashes",
        "crash", "password", "login", "payment", "refund", "fraud",
        "locked", "error", "data not syncing", "subscription", "2fa"
    ]

    full_text = (
        str(ticket["Ticket_Subject"]) + " " +
        str(ticket["Ticket_Description"])
    ).lower()

    for keyword in keyword_bank:
        if keyword in full_text:
            matched_keywords.append(keyword)

    priority_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }

    # Dossier-only simple severity estimate, grounded in input fields
    severity_estimate = 1

    if ticket["Issue_Category"] == "Fraud":
        severity_estimate += 3
    elif ticket["Issue_Category"] == "Technical":
        severity_estimate += 2
    elif ticket["Issue_Category"] == "Billing":
        severity_estimate += 1

    if ticket["Resolution_Time_Hours"] > 72:
        severity_estimate += 3
    elif ticket["Resolution_Time_Hours"] > 48:
        severity_estimate += 2
    elif ticket["Resolution_Time_Hours"] > 24:
        severity_estimate += 1

    if ticket["Satisfaction_Score"] <= 2:
        severity_estimate += 1

    severity_estimate = min(4, max(1, severity_estimate))

    reverse_priority = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Critical"
    }

    inferred_severity = reverse_priority[severity_estimate]
    severity_delta = severity_estimate - priority_map[ticket["Priority_Level"]]

    if prediction_text == "Consistent":
        mismatch_type = "Consistent"
    elif severity_delta > 0:
        mismatch_type = "Hidden Crisis"
    elif severity_delta < 0:
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Mismatch"

    dossier = {
        "ticket_id": ticket["Ticket_ID"],
        "assigned_priority": ticket["Priority_Level"],
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
                "value": f'{ticket["Resolution_Time_Hours"]} hours',
                "interpretation": (
                    "Long resolution time suggests higher severity"
                    if ticket["Resolution_Time_Hours"] > 48
                    else "Resolution time does not indicate severe delay"
                )
            },
            {
                "signal": "issue_category",
                "value": ticket["Issue_Category"],
                "interpretation": "Issue category is used as a structured severity signal"
            },
            {
                "signal": "ticket_channel",
                "value": ticket["Ticket_Channel"],
                "interpretation": "Ticket channel is used as structured metadata"
            }
        ],
        "constraint_analysis": (
            "This evidence dossier is grounded only in fields from the input ticket: "
            "subject, description, issue category, ticket channel, resolution time, "
            "assigned priority, and satisfaction score."
        ),
        "confidence": round(float(confidence), 3)
    }

    return dossier


# Single ticket audit
if st.sidebar.button("Audit Ticket"):
    ticket = {
        "Ticket_ID": ticket_id,
        "Ticket_Subject": subject,
        "Ticket_Description": description,
        "Issue_Category": issue_category,
        "Priority_Level": priority,
        "Ticket_Channel": channel,
        "Resolution_Time_Hours": resolution_time,
        "Satisfaction_Score": satisfaction
    }

    ticket_df = pd.DataFrame([ticket])
    preds, probs = predict(ticket_df)

    pred = int(preds[0])
    confidence = probs[0][pred]

    st.subheader("Single Ticket Audit Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Binary Judgment", "Mismatch" if pred == 1 else "Consistent")
    c2.metric("Confidence", round(float(confidence), 3))
    c3.metric("Assigned Priority", priority)

    dossier = create_dossier(ticket, pred, confidence)

    st.subheader("Evidence Dossier")
    st.json(dossier)

st.divider()

# Batch upload
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_preds, batch_probs = predict(batch_df)

    batch_df["Prediction_Label"] = batch_preds
    batch_df["Prediction"] = [
        "Mismatch" if pred == 1 else "Consistent"
        for pred in batch_preds
    ]

    batch_df["Confidence"] = [
        round(float(batch_probs[i][pred]), 3)
        for i, pred in enumerate(batch_preds)
    ]

    dossiers = []
    for i, row in batch_df.iterrows():
        dossiers.append(
            create_dossier(row.to_dict(), int(batch_preds[i]), batch_df.loc[i, "Confidence"])
        )

    batch_df["Evidence_Dossier"] = [str(d) for d in dossiers]

    st.subheader("Batch CSV Audit Results")
    st.dataframe(batch_df.head(50), width="stretch")

    csv_out = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Audited CSV",
        csv_out,
        "audited_tickets.csv",
        "text/csv"
    )

st.divider()

# Required dashboard charts
st.subheader("Flagged vs Consistent Tickets")

flag_df = pd.DataFrame({
    "Status": ["Consistent", "Mismatch"],
    "Count": [
        int(len(df) - df["Mismatch_Label"].sum()),
        int(df["Mismatch_Label"].sum())
    ]
})

fig_flag = px.bar(flag_df, x="Status", y="Count")
st.plotly_chart(fig_flag, width="stretch")

left, right = st.columns(2)

with left:
    st.subheader("Mismatch Type Distribution")
    mismatch_df = df["Mismatch_Type"].value_counts().reset_index()
    mismatch_df.columns = ["Mismatch_Type", "Count"]
    fig1 = px.bar(mismatch_df, x="Mismatch_Type", y="Count")
    st.plotly_chart(fig1, width="stretch")

with right:
    st.subheader("Assigned Priority Distribution")
    fig2 = px.pie(df, names="Priority_Level", title="Assigned Priority Split")
    st.plotly_chart(fig2, width="stretch")

st.subheader("Top Contributing Signals")

signal_scores = {
    "Category Score": df["Category_Score"].mean(),
    "Resolution Score": df["Resolution_Score"].mean(),
    "Text Urgency Score": df["Text_Urgency_Score"].mean()
}

signal_df = pd.DataFrame(signal_scores.items(), columns=["Signal", "Average Score"])
fig_signal = px.bar(signal_df, x="Signal", y="Average Score")
st.plotly_chart(fig_signal, width="stretch")

st.subheader("Severity Delta Heatmap Across Categories and Channels")

heatmap = pd.pivot_table(
    df,
    values="Severity_Delta",
    index="Issue_Category",
    columns="Ticket_Channel",
    aggfunc="mean"
)

fig_heat = px.imshow(
    heatmap,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(fig_heat, width="stretch")

st.subheader("Dataset Preview")
st.dataframe(df.head(30), width="stretch")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Pseudo-Labeled Dataset",
    csv,
    "pseudo_labeled_tickets.csv",
    "text/csv"
)
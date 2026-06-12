import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("models/sia_baseline_model.pkl")

st.set_page_config(
    page_title="Support Integrity Auditor",
    layout="wide"
)

st.title("Support Integrity Auditor")
st.caption("Self-supervised CRM ticket priority mismatch detection")

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

# KPI cards
total_tickets = len(df)
mismatch_count = df["Mismatch_Label"].sum()
mismatch_rate = round((mismatch_count / total_tickets) * 100, 2)
hidden_crisis = (df["Mismatch_Type"] == "Hidden Crisis").sum()
false_alarm = (df["Mismatch_Type"] == "False Alarm").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tickets", total_tickets)
col2.metric("Mismatch Rate", f"{mismatch_rate}%")
col3.metric("Hidden Crises", hidden_crisis)
col4.metric("False Alarms", false_alarm)

st.divider()

# Sidebar form
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
resolution_time = st.sidebar.number_input(
    "Resolution Time Hours",
    min_value=0,
    value=65
)
satisfaction = st.sidebar.slider(
    "Satisfaction Score",
    min_value=1,
    max_value=5,
    value=1
)

def predict(ticket):
    temp = pd.DataFrame([ticket])
    temp["Combined_Text"] = temp["Ticket_Subject"] + " " + temp["Ticket_Description"]

    X = temp[[
        "Combined_Text",
        "Issue_Category",
        "Ticket_Channel",
        "Resolution_Time_Hours",
        "Satisfaction_Score"
    ]]

    pred = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][pred]
    return pred, confidence

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

    pred, confidence = predict(ticket)

    st.subheader("Single Ticket Audit Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "Mismatch" if pred == 1 else "Consistent")
    c2.metric("Confidence", round(float(confidence), 3))
    c3.metric("Assigned Priority", priority)

    dossier = {
        "ticket_id": ticket_id,
        "assigned_priority": priority,
        "prediction": "Mismatch" if pred == 1 else "Consistent",
        "feature_evidence": [
            {"signal": "issue_category", "value": issue_category},
            {"signal": "resolution_time", "value": f"{resolution_time} hours"},
            {"signal": "satisfaction_score", "value": satisfaction}
        ],
        "confidence": round(float(confidence), 3)
    }

    st.json(dossier)

st.divider()

# Dashboard charts
left, right = st.columns(2)

with left:
    st.subheader("Mismatch Type Distribution")
    fig1 = px.bar(
        df["Mismatch_Type"].value_counts().reset_index(),
        x="Mismatch_Type",
        y="count",
        labels={"count": "Tickets"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Priority Distribution")
    fig2 = px.pie(
        df,
        names="Priority_Level",
        title="Assigned Priority Split"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Severity Heatmap: Category vs Priority")
heatmap_data = pd.crosstab(df["Issue_Category"], df["Priority_Level"])
fig3 = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Dataset Preview")
st.dataframe(df.head(30), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Pseudo-Labeled Dataset",
    csv,
    "pseudo_labeled_tickets.csv",
    "text/csv"
)
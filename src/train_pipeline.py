import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

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

y = df["Mismatch_Label"]

text_features = "Combined_Text"
categorical_features = ["Issue_Category", "Ticket_Channel"]
numeric_features = ["Resolution_Time_Hours", "Satisfaction_Score"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Training completed.")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(pipeline, "data/sia_baseline_model.pkl")
print("\nModel saved: data/sia_baseline_model.pkl")
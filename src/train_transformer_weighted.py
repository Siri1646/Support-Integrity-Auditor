import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "distilbert-base-uncased"

df = pd.read_csv("data/pseudo_labeled_tickets.csv")

df["text"] = (
    df["Ticket_Subject"].fillna("") + " " +
    df["Ticket_Description"].fillna("") + " " +
    "Category: " + df["Issue_Category"].fillna("") + " " +
    "Channel: " + df["Ticket_Channel"].fillna("") + " " +
    "Resolution hours: " + df["Resolution_Time_Hours"].astype(str) + " " +
    "Satisfaction score: " + df["Satisfaction_Score"].astype(str)
)

df = df[["text", "Mismatch_Label"]].rename(columns={"Mismatch_Label": "label"})

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="models/distilbert_sia_weighted",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=-1)
y_true = predictions.label_ids

print("\nFinal Evaluation - Weighted DistilBERT")
print("--------------------------------------")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

trainer.save_model("models/distilbert_sia_weighted/final_model")
tokenizer.save_pretrained("models/distilbert_sia_weighted/final_model")

print("\nWeighted fine-tuned DistilBERT model saved to models/distilbert_sia_weighted/final_model")
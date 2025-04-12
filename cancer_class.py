import pandas as pd
import torch
import accelerate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

df = pd.read_csv('/content/cleaned_csv.csv')

df["text"] = df["title"] + " " + df["body"]
df["label_encoded"] = df["label"].map({"Non-Cancer": 0, "Cancer": 1})


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, stratify=df["label_encoded"], random_state=42
)

# === Tokenization ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

# === Convert to Hugging Face Dataset ===
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels.tolist()
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels.tolist()
})

# === Load Model ===
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

# === Define Metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# === Train ===
trainer.train()

# === Evaluate ===
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)

print("\nClassification Report:")
print(classification_report(test_labels, preds, target_names=["Non-Cancer", "Cancer"]))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, preds))

# === Evaluate ===
preds_output = trainer.predict(train_dataset)
preds = np.argmax(preds_output.predictions, axis=1)

print("\nClassification Report:")
print(classification_report(train_labels, preds, target_names=["Non-Cancer", "Cancer"]))
print("Confusion Matrix:")
print(confusion_matrix(train_labels, preds))


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

# LoRA-specific imports
from peft import get_peft_model, LoraConfig, TaskType

# === Prepare Data ===
df["text"] = df["title"] + " " + df["body"]
df["label_encoded"] = df["label"].map({"Non-Cancer": 0, "Cancer": 1})

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, stratify=df["label_encoded"], random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels.tolist()
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels.tolist()
})

# === Load Base Model ===
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # Specific to DistilBERT
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results_lora",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

# === Compute Metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# === Train with LoRA ===
trainer.train()

# === Evaluate on Test Set ===
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
print("\nClassification Report (Test Set):")
print(classification_report(test_labels, preds, target_names=["Non-Cancer", "Cancer"]))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, preds))

# === Evaluate on Train Set ===
preds_output_train = trainer.predict(train_dataset)
preds_train = np.argmax(preds_output_train.predictions, axis=1)
print("\nClassification Report (Train Set):")
print(classification_report(train_labels, preds_train, target_names=["Non-Cancer", "Cancer"]))
print("Confusion Matrix:")
print(confusion_matrix(train_labels, preds_train))

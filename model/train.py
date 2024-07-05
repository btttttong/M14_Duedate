import os
import json
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import evaluate
import spacy
from spacy.training import offsets_to_biluo_tags

nlp = spacy.load("en_core_web_sm")

# Load the dataset
dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual")
df = pd.DataFrame(dataset["train"])

# Split the data
train_df, eval_df = train_test_split(df, test_size=0.1)

# Define label list with BIO scheme
label_list = ["O", "B-DATE", "I-DATE"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


# Tokenize and align labels
def tokenize_and_convert_to_bio(examples):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["generated_text"])):
        text = examples["generated_text"][i]
        doc = nlp(text)
        tokens = [token.text for token in doc]
        spans = json.loads(examples["pii_spans"][i])
        entities = [
            (span["start"], span["end"], span["label"])
            for span in spans
            if span["label"] == "date"
        ]

        # Manage overlapping entities
        entities = sorted(entities, key=lambda x: x[0])
        new_entities = []
        max_end = 0

        for entity in entities:
            start, end, label = entity
            if start >= max_end:
                new_entities.append(entity)
                max_end = end

        # Convert to BIO tags
        bio_tags = offsets_to_biluo_tags(doc, new_entities)
        bio_tags = [
            tag.replace("L-", "I-").replace("U-", "B-") if tag != "O" else tag
            for tag in bio_tags
        ]

        # Convert BIO tags to label ids
        label_ids = []
        for tag in bio_tags:
            if tag == "O":
                label_ids.append(0)
            elif tag.startswith("B-"):
                label_ids.append(1)
            elif tag.startswith("I-"):
                label_ids.append(2)
            else:
                label_ids.append(-100)

        # Tokenize using Hugging Face tokenizer
        tokenized_input = tokenizer(
            tokens, truncation=True, padding="max_length", is_split_into_words=True
        )

        # Align labels with tokens
        word_ids = tokenized_input.word_ids()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label_ids[word_id])

        tokenized_inputs["input_ids"].append(tokenized_input["input_ids"])
        tokenized_inputs["attention_mask"].append(tokenized_input["attention_mask"])
        tokenized_inputs["labels"].append(aligned_labels)

    return tokenized_inputs


# Convert DataFrame to Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize and align labels
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
tokenized_train_dataset = train_dataset.map(tokenize_and_convert_to_bio, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_convert_to_bio, batched=True)

# Load pre-trained model with correct number of labels
model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-multilingual-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Initialize the seqeval metric
seqeval = evaluate.load("seqeval")


# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="my_ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    no_cuda=True,  # Disable GPU usage
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model_path = "./model/ner_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

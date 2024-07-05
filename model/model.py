from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path


def load_ner_model(model_path):
    # Load the tokenizer and model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    return ner_pipeline


def load_external_ner_model(model_name):
    # Load a different NER model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    return ner_pipeline

import streamlit as st
from PIL import Image
from pathlib import Path
from model.model import load_ner_model
from utils.text_extraction import (
    extract_text_from_image,
    extract_text_from_pdf,
    extract_due_dates,
)


# Function to handle file upload and processing
def process_file(file):
    file_extension = Path(file.name).suffix.lower()
    if file_extension in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file)
        ocr_text = extract_text_from_image(image)
    elif file_extension == ".pdf":
        # Save the uploaded PDF to a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(file.read())
        ocr_text = extract_text_from_pdf("temp.pdf")
    else:
        st.error("Unsupported file format.")
        return None
    return ocr_text


# Streamlit app layout
st.title("Invoice Due Date Extractor")
st.write("Upload an invoice image or PDF to extract due dates.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
    # Process the uploaded file
    ocr_text = process_file(uploaded_file)
    if ocr_text:
        st.write("OCR Text:")
        st.write(ocr_text)

        # Load NER model
        model_path = Path("model/ner_model")  # Ensure this path is correct
        ner_pipeline = load_ner_model(model_path)

        # Extract due dates
        st.write("Extracting due dates...")
        due_dates = extract_due_dates(ocr_text, ner_pipeline)
        st.write("Due Dates Found:")
        for date in due_dates:
            st.write(date)

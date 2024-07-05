from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import re
from datetime import datetime, timedelta


def extract_text_from_image(image):
    ocr_text = pytesseract.image_to_string(image, lang="eng")
    return ocr_text


def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_text = ""
    for image in images:
        ocr_text += extract_text_from_image(image) + "\n"
    return ocr_text


def calculate_due_date(reference_date, text):
    match = re.search(r"within (\d+) days", text, re.IGNORECASE)
    if match:
        days = int(match.group(1))
        due_date = reference_date + timedelta(days=days)
        return due_date.strftime("%Y-%m-%d")
    return None


def extract_due_dates(ocr_text, ner_pipeline):
    predictions = ner_pipeline(ocr_text)
    dates = []
    document_date = None

    for prediction in predictions:
        if prediction["entity_group"] == "DATE":
            start = prediction["start"]
            end = prediction["end"]
            entity_text = ocr_text[start:end]
            dates.append(entity_text)

    invoice_date_match = re.search(r"Invoice Date:\s*([\d/]+)", ocr_text)
    if invoice_date_match:
        invoice_date_str = invoice_date_match.group(1)
        document_date = (
            datetime.strptime(invoice_date_str, "%d/%m/%Y")
            if "/" in invoice_date_str
            else datetime.strptime(invoice_date_str, "%Y-%m-%d")
        )
        dates.append(f"Invoice Date: {document_date.strftime('%Y-%m-%d')}")

    due_date_match = re.search(r"Due Date:\s*([\d/]+)", ocr_text)
    if due_date_match:
        due_date_str = due_date_match.group(1)
        due_date = (
            datetime.strptime(due_date_str, "%d/%m/%Y")
            if "/" in due_date_str
            else datetime.strptime(due_date_str, "%Y-%m-%d")
        )
        dates.append(f"Due Date: {due_date.strftime('%Y-%m-%d')}")

    if document_date:
        match = re.search(r"within (\d+) days", ocr_text, re.IGNORECASE)
        if match:
            days = int(match.group(1))
            due_date_from_relative = document_date + timedelta(days=days)
            dates.append(
                f"Calculated Due Date (from relative expression): {due_date_from_relative.strftime('%Y-%m-%d')}"
            )

    payment_due_match = re.search(r"Payment Due\s*([\w\s,]+)", ocr_text, re.IGNORECASE)
    if payment_due_match:
        due_date_str = payment_due_match.group(1).strip()
        try:
            due_date = datetime.strptime(due_date_str, "%B %d, %Y")
            dates.append(f"Payment Due Date: {due_date.strftime('%Y-%m-%d')}")
        except ValueError:
            pass

    unique_dates = list(set(dates))
    return unique_dates

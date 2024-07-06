# Invoice Due Date Extractor

This project extracts due dates from invoice images using OCR and NER models.

## Project Structure

/project_directory
    /data
    # Place your data files here
    /model
        model.py
        train.py
    main.py
    README.md
    requirements.txt

## Requirements

- Python 3.6+
- Streamlit
- pytesseract
- pillow
- transformers
- torch
- datasets
- scikit-learn

## Installation

1. **Clone the repository**
    ```sh
    git clone [https://github.com/yourusername/your_project_directory.git](https://github.com/btttttong/M14_Duedate.git)
    cd to your_project_directory
    ```

2. **Create and activate a virtual environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

4. **Check trained model in the `model` directory**:
    - The model should be saved in the directory `./model/ner_model`.

## Usage

1. **Train the Model** (If you need to train a new model):
    ```sh
    python model/train.py
    ```

2. **Run the Streamlit app**:
    ```sh
    streamlit run main.py
    ```

3. **Access the App**:
    - Open your web browser and go to `http://localhost:8501`.

4. **Upload an invoice image and extract the due dates**:
    - Use the file uploader in the app to upload an image of an invoice.
    - The app will display the OCR text and extract due dates using the NER model.

## Project Files

### `model/model.py`

This file contains functions to load and save the NER model.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_ner_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return pipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple", task='ner')

def save_ner_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

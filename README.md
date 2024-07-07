# Invoice Due Date Extractor

This project  aim to extracts due dates from invoice images using OCR and NER models.
for model using google-bert/bert-base-multilingual-cased model with "gretelai/synthetic_pii_finance_multilingual as pre-train set.

## Directory Structure
- `/data`: Contains sample invoice images and PDFs used for the project.
- `/model`: Scripts and files needed for launching the model.
- `/docs`: Project report and system design document.
- `/utils`: Utility scripts for text extraction and other helper functions.
- `main.py`: Main Python file for the Streamlit app.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of required Python packages.

## Requirements

- Python 3.6+
- Streamlit
- pytesseract
- pillow
- transformers
- torch
- datasetsS
- scikit-learn

## Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/btttttong/M14_Duedate.git
    cd your_project_directory
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

4. **Install Tesseract**
    - **Ubuntu**:
        ```sh
        sudo apt update
        sudo apt install tesseract-ocr
        sudo apt install libtesseract-dev
        ```
    - **Windows**:
        - Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
        - Follow the installation instructions.

5. **Download and place your trained model in the `model` directory**:
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


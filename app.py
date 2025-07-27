import platform
import csv
from datetime import datetime
from flask import Flask, render_template, request
import joblib
import re
from langdetect import detect
from googletrans import Translator
import string
import nltk
import os
import uuid
from PIL import Image
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
translator = Translator()

# Load ML model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# CSV setup
CSV_FILE = "submitted_messages.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "message", "prediction"])

# Text Cleaning Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_text(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, dest='en').text
    except:
        pass
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = ""
    image_path = None

    # If image is uploaded
    if 'image' in request.files and request.files['image'].filename != '':
        image = request.files['image']
        try:
            from io import BytesIO
            image_data = Image.open(BytesIO(image.read()))
            extracted_text = pytesseract.image_to_string(image_data)
            message = extracted_text.strip()
            if not message:
                return render_template('result.html',
                                       label="Image has no readable text.",
                                       confidence=0,
                                       message="Unable to extract text from image.",
                                       image_path=None)
        except Exception as e:
            return render_template('result.html',
                                   label="Image processing failed.",
                                   confidence=0,
                                   message="Could not read the uploaded image.",
                                   image_path=None)
    else:
        # If message text is entered
        message = request.form['message'].strip()

    # Final check: if still no message
    if not message:
        return render_template('result.html',
                               label="No valid text found.",
                               confidence=0,
                               message="Please enter a message or upload an image.")

    # Preprocess and predict
    processed_msg = preprocess_text(message)
    vectorized_input = vectorizer.transform([processed_msg])
    prediction = model.predict(vectorized_input)[0]
    # confidence = model.predict_proba(vectorized_input).max() * 100

    # Save to CSV
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), message, prediction])

    return render_template('result.html',
                       label=prediction.capitalize(),
                       message=message,
                       image_path=None)


# Start the app
if __name__ == '__main__':
    app.run(debug=True)

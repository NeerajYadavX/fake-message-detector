from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import pandas as pd
import os
import pickle
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message_text = request.form.get('message')
    image_file = request.files.get('image')

    extracted_text = ""

    if message_text:
        extracted_text = message_text
    elif image_file:
        image_path = "temp_image.png"
        image_file.save(image_path)
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image, lang='eng+hin')
        os.remove(image_path)

    # Predict and calculate probability
    input_vector = vectorizer.transform([extracted_text])
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector)[0]
    confidence = round(np.max(proba) * 100, 2)
    result = "Fake" if prediction == 1 else "Real"

    # Save submitted message with predicted label
    new_entry = pd.DataFrame([[extracted_text, result]], columns=['message', 'label'])
    new_entry.to_csv("user_messages.csv", mode='a', header=not os.path.exists("user_messages.csv"), index=False)

    return render_template("result.html", result=result, extracted_text=extracted_text, confidence=confidence)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


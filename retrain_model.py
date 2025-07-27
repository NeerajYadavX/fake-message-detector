import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from langdetect import detect
from googletrans import Translator
from nltk.corpus import stopwords
import nltk
import os

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
translator = Translator()

# Preprocess function
def preprocess_text(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, src=lang, dest='en').text
    except:
        pass

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load datasets
main_path = "messages_dataset.csv"
user_path = "user_messages.csv"
large_path = "fake_message_dataset_large.csv"

if not os.path.exists(main_path):
    raise FileNotFoundError(f"Missing dataset: {main_path}")
if not os.path.exists(large_path):
    raise FileNotFoundError(f"Missing dataset: {large_path}")
if not os.path.exists(user_path):
    print("Note: No user-submitted messages found. Using only base datasets.")

df_main = pd.read_csv(main_path)
df_user = pd.read_csv(user_path) if os.path.exists(user_path) else pd.DataFrame(columns=["message", "label"])
df_large = pd.read_csv(large_path)

for df in [df_main, df_user, df_large]:
    df.columns = df.columns.str.strip().str.lower()
    df['label'] = df['label'].astype(str).str.lower()

# Combine
df = pd.concat([df_main, df_user, df_large], ignore_index=True)

# Keep only valid labels
df = df[df['label'].isin(['real', 'fake', 'suspicious'])]

# Drop missing
df = df.dropna(subset=['message', 'label'])

# Clean
df['cleaned_message'] = df['message'].astype(str).apply(preprocess_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Retrained model accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Updated model and vectorizer saved successfully.")

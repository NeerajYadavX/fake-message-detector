import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load original dataset
df_original = pd.read_csv("messages_dataset.csv")
df_original['label'] = df_original['label'].map({'Fake': 1, 'Real': 0})

# Load new user messages
df_new = pd.read_csv("user_messages.csv")
df_new['label'] = df_new['label'].map({'Fake': 1, 'Real': 0})

# Combine both datasets
df_combined = pd.concat([df_original, df_new], ignore_index=True)

# Prepare data
X = df_combined['message']
y = df_combined['label']

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model retrained and saved.")

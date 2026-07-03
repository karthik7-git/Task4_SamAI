import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data components
# Download necessary NLTK data components
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------
# 1. REAL DATASET LOADING (Updated Section)
# ---------------------------------------------------------------------
print("Loading dataset...")
# Make sure your downloaded CSV file is named exactly 'news_data.csv'
df = pd.read_csv('WELFake_Dataset.csv')

# Clean out any empty rows
df = df.dropna(subset=['text', 'label'])

# Check if the labels are numbers (0 and 1) instead of text ('Real'/'Fake')
# If your dataset uses 1 for Fake and 0 for Real, uncomment the line below:
# df['label'] = df['label'].map({1: 'Fake', 0: 'Real'})

print(f"Dataset loaded successfully with {len(df)} rows!")

# ---------------------------------------------------------------------
# 2. NLP TEXT PREPROCESSING
# ---------------------------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

print("--- Preprocessing Text Data (This might take a moment...) ---")
df['clean_text'] = df['text'].apply(preprocess_text)

# ---------------------------------------------------------------------
# 3. SPLITTING AND FEATURE EXTRACTION (TF-IDF)
# ---------------------------------------------------------------------
X = df['clean_text']
y = df['label']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------------------
# 4. TRAIN AND EVALUATE ML MODEL
# ---------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\n" + "="*50)
print("--- Model Performance Metrics ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("="*50 + "\n")

# ---------------------------------------------------------------------
# 5. PREDICTION WITH CONFIDENCE SCORE FUNCTION
# ---------------------------------------------------------------------
def predict_news(news_article):
    cleaned_article = preprocess_text(news_article)
    vectorized_article = vectorizer.transform([cleaned_article])
    
    prediction = model.predict(vectorized_article)[0]
    probabilities = model.predict_proba(vectorized_article)[0]
    
    class_idx = np.where(model.classes_ == prediction)[0][0]
    confidence_score = probabilities[class_idx] * 100
    
    print("--- Live Prediction Result ---")
    print(f"Article: \"{news_article}\"")
    print(f"Prediction: {prediction}")
    print(f"Confidence Score: {confidence_score:.2f}%")
    print("-" * 30)

# Test custom statements
sample_news = "The government announced a new policy regarding university examinations today."
predict_news(sample_news)
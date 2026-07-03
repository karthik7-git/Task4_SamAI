An end-to-end Python pipeline designed to classify news articles as Real or Fake. This project leverages Natural Language Processing (NLP) for text preprocessing and applies a Machine Learning classifier to evaluate articles, displaying predictions alongside explicit confidence scores.

🚀 Features
Text Preprocessing: Automated NLP pipeline handling tokenization, lowercasing, punctuation cleaning, and stop-word filtering using NLTK.

Feature Extraction: Text conversion into numerical representations utilizing Term Frequency-Inverse Document Frequency (TF-IDF).

Classification Engine: Implements a stable supervised machine learning model (LogisticRegression) optimized for robust binary probability scoring.

Confidence Metrics: Computes and displays the exact percentage confidence score for individual article evaluations.

Performance Evaluation: Outputs clean metrics including Model Accuracy, Precision, Recall, F1-score, and a Confusion Matrix.

📂 Dataset
The model is designed to work with large-scale benchmarks such as the WELFake Dataset from Kaggle (containing over 72,000 text records).

💡 Note: Ensure your downloaded dataset file is named WELFake_Dataset.csv and placed directly inside the root directory alongside the script. Make sure it is extracted as a .csv file and not left wrapped in an unzipped folder structure!

🛠️ Installation & Environment Setup
Ensure you have a clean, stable environment setup to prevent typical library dependency conflicts on Windows systems:

Bash
# Install specific matching dependencies to ensure environment stability
pip install "numpy>=1.24,<2.2" "pandas<3.0" "scikit-learn>=1.2,<1.7" nltk
Running the Project
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
Run the main evaluation file:

Bash
python Task4.py
📈 Sample Console Output
Plaintext
Loading dataset...
Dataset loaded successfully with 72095 rows!
--- Preprocessing Text Data (This might take a moment...) ---

==================================================
--- Model Performance Metrics ---
Accuracy Score: 94.62%

Classification Report:
              precision    recall  f1-score   support
        Fake       0.95      0.94      0.94      7050
        Real       0.94      0.95      0.95      7369

==================================================

--- Live Prediction Result ---
Article: "The government announced a new policy regarding university examinations today."
Prediction: Real
Confidence Score: 89.41%
------------------------------
🛠️ Built With
Python 3.11+

NLTK - Natural Language Toolkit

Scikit-Learn - Machine Learning Library

Pandas - Data Analysis Pipeline

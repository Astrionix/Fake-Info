import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt') # Tokenizer data

# ==========================================
# STAGE 2: Dataset Selection
# ==========================================
def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    The dataset contains news articles labeled as FAKE or REAL.
    """
    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        # Dropping rows with null values if any
        df.dropna(subset=['text', 'label'], inplace=True)
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

# ==========================================
# STAGE 3: Data Preprocessing (NLP)
# ==========================================
def preprocess_text(text):
    """
    Applies NLP preprocessing techniques:
    1. Lowercasing
    2. Removing punctuation & numbers
    3. Tokenization (implicit in split or robust tokenizer)
    4. Stopword removal
    5. Lemmatization
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing punctuation & numbers (keeping only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization
    tokens = text.split()
    
    # 4. Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# ==========================================
# STAGE 4: Feature Engineering
# ==========================================
# We will use TF-IDF Vectorizer ensuring text is converted to numerical vectors.
# TF-IDF (Term Frequency-Inverse Document Frequency) captures the importance of words.

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- STAGE 2 Loading ---
    df = load_data('fake_or_real_news.csv')
    
    if df is not None:
        # --- STAGE 3 Preprocessing ---
        print("Starting data preprocessing (this may take a while)...")
        # Apply preprocessing to the 'text' column
        df['clean_text'] = df['text'].apply(preprocess_text)
        print("Data preprocessing complete.")
        
        # --- STAGE 4 Feature Engineering ---
        print("Vectorizing text data using TF-IDF...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for performance
        X = tfidf_vectorizer.fit_transform(df['clean_text']).toarray()
        y = df['label']
        print("Feature engineering complete. X shape:", X.shape)
        
        # --- STAGE 5 Model Development ---
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Training Logistic Regression Classifier...")
        # Logistic Regression is suitable for binary classification tasks on high-dimensional sparse data like text.
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        print("Model training complete.")
        
        # --- STAGE 6 Model Evaluation ---
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='REAL') # Assuming REAL is positive class or specify
        recall = recall_score(y_test, y_pred, pos_label='REAL')
        f1 = f1_score(y_test, y_pred, pos_label='REAL')
        
        print(f"\n--- Model Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (REAL): {precision:.4f}")
        print(f"Recall (REAL): {recall:.4f}")
        print(f"F1 Score (REAL): {f1:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # --- STAGE 8 Visualization ---
        print("Generating visualizations...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('reports/confusion_matrix.png')
        print("Confusion matrix saved to reports/confusion_matrix.png.")
        
        # --- Saving Model for Stage 7/Prediction System ---
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        print("Model and Vectorizer saved to models/ directory.")


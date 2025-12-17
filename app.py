from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load Model and Vectorizer
print("Loading model and vectorizer...")
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tfidf_vectorizer = None

# Preprocessing Function (Must match training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tfidf_vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    text_input = data.get('text', '')
    
    if not text_input:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess
    clean_text = preprocess_text(text_input)
    
    # Vectorize
    vectorized_text = tfidf_vectorizer.transform([clean_text]).toarray()
    
    # Predict
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    class_labels = model.classes_
    confidence = max(probabilities) * 100
    
    result = "Misinformation" if prediction == "FAKE" else "Legitimate Information"
    
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

# Page Config
st.set_page_config(
    page_title="TruthSeeker AI | Fake News Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding-bottom: 20px;
    }

    /* Text Area */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        padding: 0.6rem !important;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        background-color: #2563eb !important;
        transform: translateY(-2px);
    }
    
    /* Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .fake-news {
        background-color: rgba(239, 68, 68, 0.2);
        border-color: #ef4444;
    }
    
    .real-news {
        background-color: rgba(34, 197, 94, 0.2);
        border-color: #22c55e;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 10px;
        display: block;
    }

</style>
""", unsafe_allow_html=True)

# --- Logic setup ---

# Ensure NLTK resources are available (quietly)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Load Model & Vectorizer from cache
@st.cache_resource
def load_model():
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- UI Components ---

st.markdown("<h1>🕵️ TruthSeeker AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 30px;'>Advanced Machine Learning Misinformation Detection System</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    if model is None:
        st.error("⚠️ Model not found! Please run 'train_model.py' first.")
    else:
        news_text = st.text_area("Analyze News Article", height=200, placeholder="Paste the content of the news article here...")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze = st.button("🔍 Analyze Veracity")
        with col_btn2:
            clear = st.button("🗑️ Clear")

        if clear:
             st.rerun()

        if analyze and news_text:
            with st.spinner("Analyzing text patterns, grammar, and sourcing..."):
                time.sleep(0.8) # Simulate processing for UX
                
                # Preprocess
                clean_text = preprocess_text(news_text)
                
                # Vectorize
                vec_text = vectorizer.transform([clean_text]).toarray()
                
                # Predict
                prediction = model.predict(vec_text)[0]
                probabilities = model.predict_proba(vec_text)[0]
                confidence = max(probabilities) * 100
                
                # Display Results
                if prediction == "FAKE":
                    st.markdown(f"""
                    <div class="result-card fake-news">
                        <h2 style="color: #fca5a5; margin:0;">🚨 Potential Misinformation</h2>
                        <p style="color: #fecaca;">This content shows patterns consistent with unreliable sources.</p>
                        <span class="confidence-text" style="color: #fca5a5;">Confidence: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card real-news">
                        <h2 style="color: #86efac; margin:0;">✅ Legitimate Information</h2>
                        <p style="color: #bbf7d0;">This content appears to be from a reliable source.</p>
                        <span class="confidence-text" style="color: #86efac;">Confidence: {confidence:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Additional Metrics (Optional expansion)
                st.write("")
                with st.expander("See Analysis Details"):
                    st.info(f"The model analyzed {len(clean_text.split())} tokens after cleaning.")
                    st.progress(int(confidence))
                    
        elif analyze and not news_text:
            st.warning("Please enter some text to analyze.")

# Sidebar for Context
with st.sidebar:
    st.markdown("### About")
    st.markdown("This system uses **Logistic Regression** trained on a dataset of **6000+ news articles**.")
    
    st.markdown("### How it works")
    st.markdown("1. **Preprocessing**: Removes noise and stopwords.")
    st.markdown("2. **TF-IDF**: Converts text to numbers.")
    st.markdown("3. **Inference**: Predicts Fake vs Real.")
    
    st.markdown("---")
    st.markdown("Developed with Streamlit & Scikit-learn.")

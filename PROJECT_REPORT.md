# Final Project Report: Fake News Detection System

## STAGE 1: Problem Definition

**Problem**: The rapid spread of misinformation on social media and news outlets poses a significant threat to public opinion, democracy, and social stability. Automated systems are required to verify the authenticity of news articles efficiently.

**Classification Task**: This is a Binary Classification problem where the input is a text document (news article) and the output is a label:
- **FAKE**: Misinformation or intentionally deceptive content.
- **REAL**: Legitimate news from verifiable sources.

**Objectives**:
1. Develop a Machine Learning pipeline to classify news articles.
2. Implement NLP preprocessing to clean and standardize text data.
3. specific Feature Engineering using TF-IDF.
4. Train a Logistic Regression model for high interpretability and efficiency.
5. Deploy the model via a user-friendly Web UI.

---

## STAGE 2: Dataset Selection

**Dataset**: We utilized the "Fake or Real News" dataset.
- **Source**: Publicly available research dataset.
- **Attributes**:
    - `id`: Unique identifier.
    - `title`: Headline of the article.
    - `text`: Main body content.
    - `label`: Class label (FAKE/REAL).
- **Loading**: Implemented using Pandas (`pd.read_csv`), ensuring efficient memory handling and easy manipulation.

---

## STAGE 3: Data Preprocessing (NLP)

Raw text data is noisy and unstructured. We implemented the following pipeline (using NLTK):
1.  **Lowercasing**: Uniforms the text case (e.g., "The" -> "the").
2.  **Removing Noise**: Eliminated special characters, punctuation, and numbers to focus on linguistic content.
3.  **Tokenization**: Splitting text into individual words (tokens).
4.  **Stopword Removal**: Filtering out common words (e.g., "is", "the", "at") that carry little semantic meaning.
5.  **Lemmatization**: Reducing words to their base form (e.g., "running" -> "run") using WordNetLemmatizer, which is superior to stemming as it considers context.

---

## STAGE 4: Feature Engineering

**Technique**: TF-IDF (Term Frequency-Inverse Document Frequency).
- **TF**: Measures how frequently a term appears in a document.
- **IDF**: Measures how important a term is (weighs down frequent terms like "said" across all documents).
- **Why TF-IDF?**: Unlike simple Bag of Words, TF-IDF assigns higher weights to unique, distinguishing words, making it ideal for classification.
- **Implementation**: `TfidfVectorizer` from Scikit-learn, limited to top 5000 features to balance performance and memory.

---

## STAGE 5: Model Development

**Split**: Data was split into Training (80%) and Testing (20%) sets using Stratified sampling to maintain class balance.
**Model**: Logistic Regression.
- **Reasoning**: Logistic Regression is a robust baseline for text classification. It works well with high-dimensional, sparse data (like TF-IDF vectors) and provides probabilistic outputs (Confidence Scores), which are crucial for this application.

---

## STAGE 6: Model Evaluation

The model was evaluated on the unseen Test set using:
- **Accuracy**: Overall correctness of the model.
- **Precision**: How many predicted "REAL" news were actually real.
- **Recall**: How many actual "REAL" news were correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Visual breakdown of True Positives, False Positives, True Negatives, and False Negatives.

*(See generated Console Output and `reports/confusion_matrix.png` for specific values)*

---

## STAGE 7: Prediction System (UI)

A web interface was developed to allow end-users to interact with the model.
- **Backend**: Flask (Python). efficient API handling.
- **Frontend**: HTML5, CSS3 (Glassmorphism design), JavaScript.
- **Features**:
    - Real-time text analysis.
    - Confidence score display.
    - Visual feedback (Green for Real, Red for Misinformation).

---

## STAGE 8: Visualization

Visualizations were generated using Matplotlib and Seaborn.
- **Confusion Matrix Heatmap**: Saved to `reports/confusion_matrix.png`. This provides an immediate visual assessment of where the model makes errors (e.g., confusing Fake for Real).

---

## STAGE 9: System Architecture

**Workflow**:
1.  **User Input** -> The user pastes text into the Web UI.
2.  **API Request** -> Text is sent to the Flask backend (`/predict`).
3.  **NLP Pipeline** -> Backend applies same Preprocessing (cleaning, lemmatization) as training.
4.  **Vectorization** -> Preprocessed text is transformed using the saved TF-IDF Vectorizer.
5.  **Inference** -> Logistic Regression model calculates probability.
6.  **Response** -> Prediction and Confidence Score sent back to UI.

---

## STAGE 10: Conclusion & Future Scope

**Conclusion**:
The developed system successfully distinguishes between Fake and Real news with significant accuracy, demonstrating the power of NLP and Linear Classifiers in text analysis.

**Limitations**:
- The dataset is static and may not reflect breaking news trends.
- Logistic Regression may miss complex semantic relationships (e.g., sarcasm).

**Future Scope**:
- **Transformer Models**: Implement BERT or RoBERTa for deep contextual understanding.
- **Live Scraping**: Integrate a news crawler to verify facts against live sources.
- **Explainability**: Use LIME or SHAP to highlight exactly *which* words triggered the decision.

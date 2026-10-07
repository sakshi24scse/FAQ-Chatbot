import json
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. Download Required NLTK Resources (Silently)
# -------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # Download punkt_tab for newer NLTK releases
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

setup_nltk()

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------------------------------------------------------------------
# 2. NLP Preprocessing Function
# -------------------------------------------------------------------
def preprocess_text(text):
    """
    Cleans raw input text by:
    1. Lowercasing
    2. Tokenizing into words
    3. Removing stop words and non-alphanumeric punctuation
    4. Reducing words to root forms via Lemmatization
    """
    # Step A: Lowercase and Tokenize
    tokens = word_tokenize(text.lower())
    
    # Step B: Filter out stop words and special characters, then lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token.isalnum() and token not in stop_words
    ]
    
    # Rejoin clean tokens into a normalized string
    return " ".join(cleaned_tokens)

# -------------------------------------------------------------------
# 3. Load FAQs Knowledge Base
# -------------------------------------------------------------------
@st.cache_data
def load_faqs():
    with open('faqs.json', 'r') as file:
        return json.load(file)

faqs = load_faqs()
faq_questions = [faq["question"] for faq in faqs]

# Preprocess all stored FAQ questions
preprocessed_faqs = [preprocess_text(q) for q in faq_questions]

# -------------------------------------------------------------------
# 4. Streamlit UI Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="College FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Student FAQ Chatbot")
st.write("Ask any question regarding courses, fees, scholarships, or library timings!")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you with college information today?"}
    ]

# Render existing chat history on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -------------------------------------------------------------------
# 5. Core Chatbot Logic & Similarity Matching
# -------------------------------------------------------------------
user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    # Display user's input in chat UI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Step 1: Preprocess user question
    cleaned_user_input = preprocess_text(user_prompt)

    # Step 2: Combine cleaned user input with stored FAQ dataset for vectorization
    corpus = preprocessed_faqs + [cleaned_user_input]

    # Step 3: Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Step 4: Calculate Cosine Similarity between User Query (last vector) & Stored FAQs
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Step 5: Find the FAQ index with the highest score
    best_match_idx = similarity_scores.argmax()
    best_match_score = similarity_scores[best_match_idx]

    # Step 6: Threshold Check (Minimum similarity threshold set to 0.2 / 20%)
    SIMILARITY_THRESHOLD = 0.20

    if best_match_score >= SIMILARITY_THRESHOLD:
        bot_response = faqs[best_match_idx]["answer"]
    else:
        bot_response = "I'm sorry, I couldn't find an answer related to your query. Please contact college support for assistance."

    # Render bot response in UI
    with st.chat_message("assistant"):
        st.write(bot_response)

    # Save bot response to session history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
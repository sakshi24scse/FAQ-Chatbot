# 🤖 Student FAQ Chatbot using NLP & TF-IDF

An interactive, natural language processing (NLP) powered chatbot designed to answer student queries regarding college admissions, hostel fees, courses, and library hours. Built using **Python**, **NLTK**, **Scikit-Learn**, and **Streamlit** as part of the CodeAlpha AI Internship program.

---

## 📌 Key Features
- **NLP Preprocessing Pipeline:** Automatic text cleaning including lowercasing, tokenization, stop-word removal, and Lemmatization via NLTK.
- **Vector Space Modeling:** Converts unstructured textual data into numerical feature matrices using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- **Semantic Matching:** Measures query-to-FAQ similarity using **Cosine Similarity**.
- **Confidence Thresholding:** Features fallback default responses when maximum similarity score falls below a set threshold ($20\%$).
- **Interactive Web Interface:** Modern, real-time chat interface rendered using Streamlit.
- **Decoupled Knowledge Base:** FAQs are stored in a structured `faqs.json` file for easy dataset updates.

---

## 🛠️ How It Works (NLP Pipeline)

```text
[ User Input Query ]
         │
         ▼
[ 1. Preprocessing ] ──► (Lowercasing ➔ Tokenization ➔ Stop-words Removal ➔ Lemmatization)
         │
         ▼
[ 2. Vectorization ] ──► (TF-IDF Vectorizer converts text into numerical vectors)
         │
         ▼
[ 3. Cosine Similarity ] ──► (Calculates dot product / distance against preprocessed FAQs)
         │
         ▼
[ 4. Score Thresholding ]
    ├── Score ≥ 0.20 ──► Output Best Matching Answer
    └── Score < 0.20 ──► Output Fallback Default Message

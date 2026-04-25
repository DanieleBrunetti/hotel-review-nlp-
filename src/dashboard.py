import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -------------------------------
# 1. CONFIGURAZIONE PAGINA
# -------------------------------
st.set_page_config(page_title="Hotel Review Analyzer", layout="centered")

st.title("🏨 Hotel Review Analyzer")
st.write("Sistema di classificazione automatica delle recensioni")

# -------------------------------
# 2. CARICAMENTO MODELLI
# -------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# -------------------------------
# 3. FUNZIONE DI PREDIZIONE
# -------------------------------
def predict(text):
    text_tfidf = vectorizer.transform([text])
    
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf).max()

    return prediction, probability

# -------------------------------
# 4. INPUT MANUALE
# -------------------------------
st.subheader("✏️ Inserisci una recensione")

user_input = st.text_area("Scrivi qui la recensione:")

if st.button("Analizza"):
    if user_input.strip() == "":
        st.warning("Inserisci una recensione valida.")
    else:
        pred, prob = predict(user_input)

        st.success("Risultato:")
        st.write(f"**Sentiment:** {pred}")
        st.write(f"**Probabilità:** {prob:.2f}")

# -------------------------------
# 5. ANALISI BATCH (CSV)
# -------------------------------
st.subheader("📂 Analisi batch da CSV")

uploaded_file = st.file_uploader("Carica un file CSV con una colonna 'body'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "body" not in df.columns:
        st.error("Il file deve contenere una colonna chiamata 'body'")
    else:
        st.write("Anteprima dati:")
        st.dataframe(df.head())

        if st.button("Analizza CSV"):
            predictions = []
            probabilities = []

            for text in df["body"]:
                pred, prob = predict(text)
                predictions.append(pred)
                probabilities.append(prob)

            df["predicted_sentiment"] = predictions
            df["confidence"] = probabilities
            df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success("Analisi completata!")
            st.dataframe(df)

            # -------------------------------
            # 6. DOWNLOAD RISULTATI
            # -------------------------------
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Scarica risultati",
                data=csv,
                file_name="risultati_analisi.csv",
                mime="text/csv"
            )

# -------------------------------
# 7. FOOTER
# -------------------------------
st.markdown("---")
st.caption("Progetto Machine Learning - Sentiment Analysis Hotel")
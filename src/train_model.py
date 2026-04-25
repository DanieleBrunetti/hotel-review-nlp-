import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model():
    """
    Carica il dataset, addestra il modello di sentiment analysis
    e salva modello e vettorizzatore.
    """

    # 1. Caricamento dataset
    df = pd.read_csv("hotel_reviews_synthetic.csv")

    # 2. Preparazione dati
    X = df["body"]          # Testo recensioni
    y = df["sentiment"]     # Etichette (positivo/negativo)

    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Vettorizzazione TF-IDF
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Addestramento modello
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # 6. Predizioni
    y_pred = model.predict(X_test_tfidf)

    # 7. Valutazione
    print("\n--- REPORT DI VALUTAZIONE ---")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Totale: {accuracy:.2f}")

    print("\nF1-Score e Metriche per Classe:")
    print(classification_report(y_test, y_pred))

    print("\nMatrice di Confusione:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Salvataggio modello e vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nModello e vectorizer salvati nella cartella 'models/'")


# 9. Esecuzione script
if __name__ == "__main__":
    train_model()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Caricamento del dataset sintetico
df = pd.read_csv('hotel_reviews_synthetic.csv')

# 2. Preparazione dei dati
# X = Testo delle recensioni (body), y = Etichetta da indovinare (sentiment)
X = df['body']
y = df['sentiment']

# 3. Train/Test Split (80% addestramento, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vettorizzazione (Il "Cervello" Digitale)
# Converte il testo in una matrice numerica TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Addestramento del Modello (Algoritmo di Classificazione)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 6. Valutazione
y_pred = model.predict(X_test_tfidf)

# --- OUTPUT DEI RISULTATI ---
print("--- REPORT DI VALUTAZIONE ---")
print(f"Accuracy Totale: {accuracy_score(y_test, y_pred):.2f}")
print("\nF1-Score e Metriche per Classe:")
print(classification_report(y_test, y_pred))

print("\nMatrice di Confusione (Semplice):")
print(confusion_matrix(y_test, y_pred))
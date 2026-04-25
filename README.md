# hotel-review-nlp-
# 🏨 Hotel Review NLP

Sistema di Smistamento Automatico delle Recensioni e Analisi del Sentiment

---

## 📌 Descrizione del Progetto

Questo progetto implementa un sistema di **Machine Learning** in grado di analizzare automaticamente recensioni di strutture ricettive (hotel, B&B) per:

* identificare il **reparto competente** (Pulizia, Reception, Ristorazione)
* determinare il **sentiment** (positivo o negativo)
* fornire risultati immediati tramite una **dashboard interattiva**

L’obiettivo è automatizzare un processo normalmente manuale, migliorando velocità, efficienza e capacità decisionale.

---

## 🎯 Obiettivi

* Smistare automaticamente le recensioni ai reparti corretti
* Analizzare il tono delle recensioni (sentiment analysis)
* Utilizzare tecniche di **Natural Language Processing (NLP)**
* Creare un prototipo **semplice, riproducibile e scalabile**
* Integrare una **interfaccia grafica** per uso pratico

---

## 🧠 Come funziona

Il sistema segue una pipeline di Machine Learning:

1. **Generazione Dataset Sintetico**

   * Creazione automatica di recensioni realistiche
   * Dataset bilanciato (positivo/negativo)
   * Inserimento di casi ambigui per test avanzati

2. **Pre-processing**

   * Separazione testo (input) e label (output)

3. **Vettorizzazione**

   * Conversione testo → numeri tramite TF-IDF

4. **Addestramento Modello**

   * Algoritmo: Logistic Regression

5. **Predizione**

   * Classificazione reparto
   * Analisi sentiment

6. **Valutazione**

   * Accuracy
   * F1-score
   * Confusion Matrix

---

## ⚙️ Tecnologie utilizzate

* Python
* pandas
* scikit-learn
* streamlit
* joblib

---

## 📂 Struttura del Progetto

```
hotel-review-nlp/
│
├── data/
│   └── hotel_reviews_synthetic.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── train_model.py
│   └── dashboard.py
│
├── models/
│   ├── vectorizer.pkl
│   ├── sentiment_model.pkl
│   └── dept_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Come eseguire il progetto

### 1. Clonare la repository

```bash
git clone https://github.com/DanieleBrunetti/hotel-review-nlp.git
cd hotel-review-nlp
```

---

### 2. Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

### 3. Generare il dataset

```bash
python src/generate_dataset.py
```

---

### 4. Addestrare il modello

```bash
python src/train_model.py
```

---

### 5. Avviare la dashboard

```bash
streamlit run src/dashboard.py
```

---

## 🖥️ Funzionalità della Dashboard

* Inserimento manuale recensione
* Predizione immediata:

  * reparto consigliato
  * sentiment
  * probabilità
* Upload CSV per analisi batch
* Export risultati con timestamp

---

## 📊 Esempio Output

Input:

```
"La camera era pulita ma il check-in è stato lentissimo"
```

Output:

```
Reparto: Reception  
Sentiment: Negativo (0.87)
```

---

## 📈 Metriche di valutazione

* **Accuracy**: ~90%+
* **F1-score**: bilanciato tra classi
* **Confusion Matrix**: analisi errori dettagliata

---

## 🔍 Limiti del sistema

* Non gestisce bene sarcasmo o ironia
* Dataset sintetico (non completamente realistico)
* Comprensione semantica limitata

---

## 🚀 Possibili sviluppi futuri

* Utilizzo di modelli NLP avanzati (es. Transformer)
* Dataset reale e più ampio
* Analisi multi-classe (es. molto positivo/negativo)
* Rilevazione emozioni
* Deploy cloud

---

## 📌 Valore del progetto

Questo progetto dimostra come trasformare:

* dati testuali non strutturati

in:

* informazioni utili e operative

Applicabile a:

* hotel
* customer service
* digital marketing
* analisi reputazione online

---

## 👨‍💻 Autore Daniele Brunetti

Project Work sviluppato per dimostrare competenze in:

* Machine Learning
* Natural Language Processing
* Data Analysis
* Sviluppo di prototipi applicativi

---

📄​ License

Vedi LICENSE per dettagli. 

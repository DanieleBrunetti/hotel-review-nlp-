import pandas as pd
import random

def generate_hotel_dataset(n_samples=100, seed=42):
    """
    Genera un dataset sintetico di recensioni hotel.
    
    Parametri:
    - n_samples: numero totale di recensioni
    - seed: per rendere i risultati riproducibili
    """

    # Imposta il seed per risultati sempre uguali
    random.seed(seed)

    # 1. Vocabolario base (reparto + sentiment)
    data_map = {
        "Pulizia": {
            "pos": [
                "La camera era davvero pulita",
                "Tutto molto ordinato e profumato",
                "Bagno impeccabile"
            ],
            "neg": [
                "La camera non era molto pulita",
                "Ho trovato polvere ovunque",
                "Il bagno lasciava a desiderare"
            ]
        },
        "Ristorazione": {
            "pos": [
                "Colazione molto varia",
                "Il cibo era davvero buono",
                "Ottima qualità dei piatti"
            ],
            "neg": [
                "Colazione piuttosto scarsa",
                "Il cibo non mi ha convinto",
                "Poca scelta nel buffet"
            ]
        },
        "Reception": {
            "pos": [
                "Personale molto cordiale",
                "Check-in rapido",
                "Staff sempre pronto ad aiutare"
            ],
            "neg": [
                "Personale poco disponibile",
                "Check-in molto lento",
                "Esperienza negativa alla reception"
            ]
        }
    }

    # Parti opzionali per rendere le frasi più realistiche
    intros = [
        "Ho soggiornato per due notti",
        "Sono stato in questo hotel per lavoro",
        "Abbiamo passato un weekend qui",
        "Prima volta in questa struttura"
    ]

    connectors = [
        "Nel complesso",
        "Devo dire che",
        "Onestamente",
        "Tutto sommato"
    ]

    endings_pos = [
        "mi sono trovato bene",
        "ci tornerei volentieri",
        "lo consiglio"
    ]

    endings_neg = [
        "non sono soddisfatto",
        "non credo che tornerò",
        "non lo consiglierei"
    ]

    rows = []

    # 2. Generazione delle recensioni
    for i in range(n_samples - 2):

        # Scelta casuale reparto e sentiment
        dept = random.choice(list(data_map.keys()))
        sentiment = random.choice(["pos", "neg"])

        # Frase principale
        main_phrase = random.choice(data_map[dept][sentiment])

        # 50% probabilità di aggiungere una seconda frase
        if random.random() > 0.5:
            other_dept = random.choice([d for d in data_map.keys() if d != dept])
            other_sent = random.choice(["pos", "neg"])
            secondary = random.choice(data_map[other_dept][other_sent])

            body_text = f"{random.choice(intros)}. {main_phrase}. {random.choice(connectors)}, {secondary}."
        else:
            body_text = f"{random.choice(intros)}. {main_phrase}."

        # Frase finale coerente con il sentiment
        ending = random.choice(endings_pos if sentiment == "pos" else endings_neg)

        full_body = f"{body_text} {ending}."

        # Costruzione riga dataset
        rows.append({
            "id": i + 1,
            "title": f"Recensione {dept}",
            "body": full_body,
            "department": dept,
            "sentiment": "positivo" if sentiment == "pos" else "negativo"
        })

    # 3. Casi ambigui (più realistici)
    ambiguous_samples = [
        {
            "id": n_samples - 1,
            "title": "Esperienza mista",
            "body": "La camera era pulita e profumata, però il check-in è stato lentissimo e snervante. Nel complesso esperienza così così.",
            "department": "Reception",
            "sentiment": "negativo"
        },
        {
            "id": n_samples,
            "title": "Bene ma migliorabile",
            "body": "Colazione ottima e abbondante, ma ho trovato della polvere sotto al letto. Mi aspettavo di meglio.",
            "department": "Pulizia",
            "sentiment": "negativo"
        }
    ]

    rows.extend(ambiguous_samples)

    # 4. Creazione DataFrame
    df = pd.DataFrame(rows)

    # 5. Salvataggio CSV
    filename = "hotel_reviews_synthetic.csv"
    df.to_csv(filename, index=False, encoding="utf-8")

    print(f"Dataset generato con successo: {len(df)} righe salvate in '{filename}'.")

    return df


# 6. Esecuzione del file
if __name__ == "__main__":
    dataset = generate_hotel_dataset(100)

    # Anteprima
    print("\nAnteprima dataset:")
    print(dataset[['department', 'sentiment', 'body']].head(5))
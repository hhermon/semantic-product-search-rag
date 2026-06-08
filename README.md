# semantic-product-search-rag

AI-driven produktupptäckt i e-handel: en prototyp som jämför nyckelordsbaserad
sökning (TF-IDF) med semantisk sökning (Sentence-BERT), och som bygger vidare
på den semantiska retrievern med ett RAG-system och ett agentbaserat system.

Källkod till examensarbetet *AI-driven produktupptäckt i e-handel* (KTH, 2026),
Hermon Asmerom & Zaid Amin.

## Innehåll

- `app.py` – Streamlit-gränssnitt för att köra sökning, RAG och agentläge interaktivt
- `evaluate.py` – kör retrieval-utvärderingen (Precision@k, Recall@k, F1, AP, MAP)
- `measure_latency.py` – mäter indexbyggnadstid och söktider per metod
- `src/` – kärnlogik (retrieval, TF-IDF, RAG-pipeline, agent, utvärderingsmått)
  - `src/evaluation/metrics.py` – måtten implementerade från grunden, utan externa IR-bibliotek
- `requirements.txt` – Python-beroenden
- `.env.example` – mall för miljövariabler (API-nyckel)
- `.streamlit/` – konfiguration för Streamlit-appen

## Förutsättningar

- Python 3.10+
- En Google Gemini API-nyckel (gratis tier räcker för experiment)

## Installation

```bash
git clone https://github.com/hhermon/semantic-product-search-rag.git
cd semantic-product-search-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Konfiguration

Kopiera `.env.example` till `.env` och fyll i er API-nyckel:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=din_nyckel_här
```

## Data

Prototypen använder **Amazon Products Dataset 2023** (Kaggle, ODC Attribution
License):
https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products

Urvalet i arbetet: 500 produkter per kategori med fast slumpfrö `seed=42`,
totalt 5 000 produkter över tio kategorier (Men's Watches, Men's Shoes,
Women's Shoes, Headphones & Earbuds, Backpacks, Skin Care Products,
Toys & Games, Men's Clothing, Luggage, Kitchen & Dining).

Ladda ner `amazon_products.csv` och `amazon_categories.csv` från Kaggle och
placera dem i `src/data/`.

## Använda systemet

Interaktivt gränssnitt:

```bash
streamlit run app.py
```

Kör retrieval-utvärderingen (genererar Precision@10, MAP m.m. för båda metoder):

```bash
python evaluate.py
```

Mät svarstider och indexbyggnadstid:

```bash
python measure_latency.py
```

Resultat sparas automatiskt i `results/`.

## Reproducerbarhet

- Slumpfrö `seed=42` används vid urval av produkter.
- Embeddingmodell: `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT, 384 dimensioner).
- Generativ modell: Gemini 2.5 Flash via `google-genai`.
- Retrieval-mått implementerade från grunden i `src/evaluation/metrics.py`.

## Licens

Datasetet används under ODC Attribution License. Övrig kod i detta repo är licensierad under MIT License.

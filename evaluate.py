"""Jämför semantisk sökning mot nyckelordssökning med Precision, Recall och MAP.

Testfrågor är grupperade i fyra kategorier:
  Grupp 1 – Kategoribaserade frågor   (direkta nyckelordsmatchningar)
  Grupp 2 – Attributbaserade frågor   (semantiska egenskaper och kombinationer)
  Grupp 3 – Intentionsbaserade frågor (användarens syfte snarare än produktnamn)
  Grupp 4 – Svenska frågor            (testar flerspråkig generalisering)

Relevansbedömning: alla produkter inom förväntad kategori anses relevanta.
Utvärderingsmått beräknade med egenimplementerade funktioner (src/evaluation/metrics.py),
inga externa IR-bibliotek används.
"""

import csv
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever import ProductRetriever, SearchResult
from src.retrieval.keyword_search import KeywordSearcher
from src.data.product_loader import ProductLoader
from src.evaluation.metrics import evaluate_query, evaluate_all

# Testfrågor med förväntad kategori som relevanskriterium.
# Relevant = alla produkter i den angivna kategorin i det stratifierade urvalet.
TEST_QUERIES = [
    # --- Grupp 1: Kategoribaserade frågor ---
    {"id": "g1q1", "group": "Kategoribaserad",   "query": "mens watch",                              "category": "Men's Watches"},
    {"id": "g1q2", "group": "Kategoribaserad",   "query": "wireless headphones",                     "category": "Headphones & Earbuds"},
    {"id": "g1q3", "group": "Kategoribaserad",   "query": "women running shoes",                     "category": "Women's Shoes"},
    {"id": "g1q4", "group": "Kategoribaserad",   "query": "backpack",                                "category": "Backpacks"},

    # --- Grupp 2: Attributbaserade frågor ---
    {"id": "g2q1", "group": "Attributbaserad",   "query": "noise cancelling earphones for office",   "category": "Headphones & Earbuds"},
    {"id": "g2q2", "group": "Attributbaserad",   "query": "luxury dress watch stainless steel",      "category": "Men's Watches"},
    {"id": "g2q3", "group": "Attributbaserad",   "query": "waterproof hiking shoes women",           "category": "Women's Shoes"},
    {"id": "g2q4", "group": "Attributbaserad",   "query": "large travel backpack carry on",          "category": "Backpacks"},

    # --- Grupp 3: Intentionsbaserade frågor ---
    {"id": "g3q1", "group": "Intentionsbaserad", "query": "gift for man who likes sports",           "category": "Men's Shoes"},
    {"id": "g3q2", "group": "Intentionsbaserad", "query": "moisturizer for dry sensitive skin",      "category": "Skin Care Products"},
    {"id": "g3q3", "group": "Intentionsbaserad", "query": "educational toy for 5 year old",          "category": "Toys & Games"},
    {"id": "g3q4", "group": "Intentionsbaserad", "query": "lightweight luggage for weekend trip",    "category": "Luggage"},

    # --- Grupp 4: Svenska frågor ---
    {"id": "g4q1", "group": "Svenska",           "query": "trådlösa hörlurar",                      "category": "Headphones & Earbuds"},
    {"id": "g4q2", "group": "Svenska",           "query": "löparskor herr",                         "category": "Men's Shoes"},
    {"id": "g4q3", "group": "Svenska",           "query": "ansiktskräm torr hud",                   "category": "Skin Care Products"},
    {"id": "g4q4", "group": "Svenska",           "query": "ryggsäck skola barn",                    "category": "Backpacks"},
]

GROUPS = ["Kategoribaserad", "Attributbaserad", "Intentionsbaserad", "Svenska"]


def keyword_to_search_results(keyword_results) -> list[SearchResult]:
    return [SearchResult(product=r.product, score=r.score, rank=r.rank) for r in keyword_results]


def main():
    print("Hämtar produkter och bygger index...")
    products = ProductLoader().load()
    retriever = ProductRetriever.from_data_file()
    keyword_searcher = KeywordSearcher.from_products(products)
    print(f"Klart: {len(products)} produkter\n")

    # Bygg kategori → produkt-ID-mängd för relevansbedömning
    category_ids: dict[str, list[str]] = {}
    for p in products:
        category_ids.setdefault(p.category, []).append(p.id)

    semantic_evals, keyword_evals = [], []

    for tq in TEST_QUERIES:
        relevant_ids = category_ids.get(tq["category"], [])
        sem_results = retriever.search(query=tq["query"], top_k=10)
        kw_results  = keyword_to_search_results(
            keyword_searcher.search(query=tq["query"], top_k=10)
        )
        sem_eval = evaluate_query(tq["id"], tq["query"], sem_results, relevant_ids)
        kw_eval  = evaluate_query(tq["id"], tq["query"], kw_results,  relevant_ids)
        sem_eval.group = tq["group"]
        kw_eval.group  = tq["group"]
        semantic_evals.append(sem_eval)
        keyword_evals.append(kw_eval)

        sem_names = [r.product.name[:35] for r in sem_results[:3]]
        kw_names  = [r.product.name[:35] for r in kw_results[:3]]
        print(f"[{tq['id']}] [{tq['group']}] {tq['query']}  (relevant: {len(relevant_ids)} produkter)")
        print(f"  Semantisk:  P={sem_eval.precision:.2f} R={sem_eval.recall:.2f} AP={sem_eval.average_precision:.2f} → {sem_names}")
        print(f"  Nyckelord:  P={kw_eval.precision:.2f} R={kw_eval.recall:.2f} AP={kw_eval.average_precision:.2f} → {kw_names}")
        print()

    sem_report = evaluate_all(semantic_evals)
    kw_report  = evaluate_all(keyword_evals)

    print("=" * 70)
    print(f"{'Metod':<22} {'MAP':>8} {'Mean P':>8} {'Mean R':>8} {'Mean F1':>8}")
    print("-" * 70)
    print(f"{'Semantisk (SBERT)':<22} {sem_report.mean_average_precision:>8.3f} {sem_report.mean_precision:>8.3f} {sem_report.mean_recall:>8.3f} {sem_report.mean_f1:>8.3f}")
    print(f"{'Nyckelord (TF-IDF)':<22} {kw_report.mean_average_precision:>8.3f} {kw_report.mean_precision:>8.3f} {kw_report.mean_recall:>8.3f} {kw_report.mean_f1:>8.3f}")
    print("=" * 70)

    print()
    for group in GROUPS:
        s_evals = [e for e in semantic_evals if e.group == group]
        k_evals = [e for e in keyword_evals  if e.group == group]
        s = evaluate_all(s_evals)
        k = evaluate_all(k_evals)
        print(f"{group}")
        print(f"  Semantisk  MAP={s.mean_average_precision:.3f}  P={s.mean_precision:.3f}  R={s.mean_recall:.3f}")
        print(f"  Nyckelord  MAP={k.mean_average_precision:.3f}  P={k.mean_precision:.3f}  R={k.mean_recall:.3f}")

    _save_csv(semantic_evals, keyword_evals, sem_report, kw_report)


def _save_csv(semantic_evals, keyword_evals, sem_report, kw_report):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    queries_path = out_dir / f"queries_{timestamp}.csv"
    with queries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "group", "query", "method", "precision", "recall", "f1", "average_precision"])
        for e in semantic_evals:
            writer.writerow([e.query_id, e.group, e.query, "semantic", f"{e.precision:.4f}", f"{e.recall:.4f}", f"{e.f1:.4f}", f"{e.average_precision:.4f}"])
        for e in keyword_evals:
            writer.writerow([e.query_id, e.group, e.query, "keyword", f"{e.precision:.4f}", f"{e.recall:.4f}", f"{e.f1:.4f}", f"{e.average_precision:.4f}"])

    summary_path = out_dir / f"summary_{timestamp}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "map", "mean_precision", "mean_recall", "mean_f1"])
        writer.writerow(["semantic", f"{sem_report.mean_average_precision:.4f}", f"{sem_report.mean_precision:.4f}", f"{sem_report.mean_recall:.4f}", f"{sem_report.mean_f1:.4f}"])
        writer.writerow(["keyword",  f"{kw_report.mean_average_precision:.4f}",  f"{kw_report.mean_precision:.4f}",  f"{kw_report.mean_recall:.4f}",  f"{kw_report.mean_f1:.4f}"])

    print(f"\nResultat sparade: {queries_path}  |  {summary_path}")


if __name__ == "__main__":
    main()

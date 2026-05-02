"""Jämför semantisk sökning mot nyckelordssökning med Precision, Recall och MAP."""

import csv
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever import ProductRetriever, SearchResult
from src.retrieval.keyword_search import KeywordSearcher
from src.data.product_loader import ProductLoader
from src.evaluation.metrics import evaluate_query, evaluate_all

TEST_QUERIES = [
    {"id": "q1", "query": "smartphone with good camera",    "relevant_ids": ["123", "130", "131", "136"]},
    {"id": "q2", "query": "laptop for programming",         "relevant_ids": ["78", "79", "80", "81", "82"]},
    {"id": "q3", "query": "men's watch luxury",             "relevant_ids": ["94", "95", "96", "97", "98", "190"]},
    {"id": "q4", "query": "wireless headphones",            "relevant_ids": ["100", "101", "107"]},
    {"id": "q5", "query": "skin care moisturizer",          "relevant_ids": ["118", "119", "120"]},
    {"id": "q6", "query": "running shoes sports sneakers",  "relevant_ids": ["88", "90", "91", "92"]},
    {"id": "q7", "query": "tablet iPad",                    "relevant_ids": ["159", "160", "161"]},
    {"id": "q8", "query": "home furniture sofa",            "relevant_ids": ["11", "12", "13", "14"]},
]


def keyword_to_search_results(keyword_results) -> list[SearchResult]:
    return [SearchResult(product=r.product, score=r.score, rank=r.rank) for r in keyword_results]


def main():
    print("Hämtar produkter och bygger index...")
    products = ProductLoader().load()
    retriever = ProductRetriever.from_data_file()
    keyword_searcher = KeywordSearcher.from_products(products)
    print(f"Klart: {len(products)} produkter\n")

    semantic_evals, keyword_evals = [], []

    for tq in TEST_QUERIES:
        sem_results = retriever.search(query=tq["query"], top_k=5)
        kw_results = keyword_to_search_results(
            keyword_searcher.search(query=tq["query"], top_k=5)
        )
        sem_eval = evaluate_query(tq["id"], tq["query"], sem_results, tq["relevant_ids"])
        kw_eval  = evaluate_query(tq["id"], tq["query"], kw_results,  tq["relevant_ids"])
        semantic_evals.append(sem_eval)
        keyword_evals.append(kw_eval)

        sem_names = [r.product.name[:25] for r in sem_results[:3]]
        kw_names  = [r.product.name[:25] for r in kw_results[:3]]
        print(f"[{tq['id']}] {tq['query']}")
        print(f"  Semantisk:  P={sem_eval.precision:.2f} R={sem_eval.recall:.2f} AP={sem_eval.average_precision:.2f} → {sem_names}")
        print(f"  Nyckelord:  P={kw_eval.precision:.2f} R={kw_eval.recall:.2f} AP={kw_eval.average_precision:.2f} → {kw_names}")
        print()

    sem_report = evaluate_all(semantic_evals)
    kw_report  = evaluate_all(keyword_evals)

    print("=" * 60)
    print(f"{'Metod':<22} {'MAP':>8} {'Mean P':>8} {'Mean R':>8} {'Mean F1':>8}")
    print("-" * 60)
    print(f"{'Semantisk (SBERT)':<22} {sem_report.mean_average_precision:>8.3f} {sem_report.mean_precision:>8.3f} {sem_report.mean_recall:>8.3f} {sem_report.mean_f1:>8.3f}")
    print(f"{'Nyckelord (TF-IDF)':<22} {kw_report.mean_average_precision:>8.3f} {kw_report.mean_precision:>8.3f} {kw_report.mean_recall:>8.3f} {kw_report.mean_f1:>8.3f}")
    print("=" * 60)

    _save_csv(semantic_evals, keyword_evals, sem_report, kw_report)


def _save_csv(semantic_evals, keyword_evals, sem_report, kw_report):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    queries_path = out_dir / f"queries_{timestamp}.csv"
    with queries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "method", "precision", "recall", "f1", "average_precision"])
        for e in semantic_evals:
            writer.writerow([e.query_id, e.query, "semantic", f"{e.precision:.4f}", f"{e.recall:.4f}", f"{e.f1:.4f}", f"{e.average_precision:.4f}"])
        for e in keyword_evals:
            writer.writerow([e.query_id, e.query, "keyword", f"{e.precision:.4f}", f"{e.recall:.4f}", f"{e.f1:.4f}", f"{e.average_precision:.4f}"])

    summary_path = out_dir / f"summary_{timestamp}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "map", "mean_precision", "mean_recall", "mean_f1"])
        writer.writerow(["semantic", f"{sem_report.mean_average_precision:.4f}", f"{sem_report.mean_precision:.4f}", f"{sem_report.mean_recall:.4f}", f"{sem_report.mean_f1:.4f}"])
        writer.writerow(["keyword",  f"{kw_report.mean_average_precision:.4f}",  f"{kw_report.mean_precision:.4f}",  f"{kw_report.mean_recall:.4f}",  f"{kw_report.mean_f1:.4f}"])

    print(f"\nResultat sparade: {queries_path}  |  {summary_path}")


if __name__ == "__main__":
    main()

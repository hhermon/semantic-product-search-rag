"""Mäter söksvarstider för semantisk retrieval och TF-IDF.

Kör varje testfråga N gånger och rapporterar genomsnitt, min och max i millisekunder.
Resultaten sparas till results/latency_<timestamp>.csv.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever import ProductRetriever
from src.retrieval.keyword_search import KeywordSearcher
from src.data.product_loader import ProductLoader

TEST_QUERIES = [
    "mens watch",
    "wireless headphones",
    "women running shoes",
    "backpack",
    "noise cancelling earphones for office",
    "luxury dress watch stainless steel",
    "waterproof hiking shoes women",
    "large travel backpack carry on",
    "gift for man who likes sports",
    "moisturizer for dry sensitive skin",
    "educational toy for 5 year old",
    "lightweight luggage for weekend trip",
    "trådlösa hörlurar",
    "löparskor herr",
    "ansiktskräm torr hud",
    "ryggsäck skola barn",
]

N_RUNS = 10  # antal körningar per fråga för stabilt medelvärde
TOP_K = 10


def measure(fn, *args, **kwargs) -> float:
    """Returnerar exekveringstid i millisekunder."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000


def main():
    print("Laddar produkter och bygger index...")

    t0 = time.perf_counter()
    products = ProductLoader().load()
    load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    retriever = ProductRetriever.from_data_file()
    sem_index_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    keyword_searcher = KeywordSearcher.from_products(products)
    kw_index_ms = (time.perf_counter() - t0) * 1000

    print(f"  Produktladdning:        {load_ms:7.1f} ms")
    print(f"  Semantiskt index (SBERT): {sem_index_ms:7.1f} ms  (embedding av 5 000 produkter)")
    print(f"  TF-IDF-index:           {kw_index_ms:7.1f} ms")
    print()

    sem_times: list[float] = []
    kw_times: list[float] = []

    print(f"Mäter söksvarstider ({N_RUNS} körningar per fråga, top_k={TOP_K})...\n")
    print(f"{'Fråga':<45} {'Sem (ms)':>10} {'KW (ms)':>10}")
    print("-" * 67)

    for query in TEST_QUERIES:
        sem_runs = [
            measure(retriever.search, query=query, top_k=TOP_K)
            for _ in range(N_RUNS)
        ]
        kw_runs = [
            measure(keyword_searcher.search, query=query, top_k=TOP_K)
            for _ in range(N_RUNS)
        ]
        sem_avg = mean(sem_runs)
        kw_avg = mean(kw_runs)
        sem_times.extend(sem_runs)
        kw_times.extend(kw_runs)
        print(f"{query:<45} {sem_avg:>10.2f} {kw_avg:>10.2f}")

    print("-" * 67)
    print(f"{'Genomsnitt (alla frågor)':<45} {mean(sem_times):>10.2f} {mean(kw_times):>10.2f}")
    print(f"{'Standardavvikelse':<45} {stdev(sem_times):>10.2f} {stdev(kw_times):>10.2f}")
    print(f"{'Min':<45} {min(sem_times):>10.2f} {min(kw_times):>10.2f}")
    print(f"{'Max':<45} {max(sem_times):>10.2f} {max(kw_times):>10.2f}")
    print()

    print("Indexbyggningstider:")
    print(f"  Semantisk (SBERT embedding, 5 000 produkter): {sem_index_ms:.0f} ms  (en gång vid uppstart)")
    print(f"  TF-IDF-index:                                 {kw_index_ms:.0f} ms  (en gång vid uppstart)")

    _save_csv(sem_times, kw_times, sem_index_ms, kw_index_ms)


def _save_csv(sem_times, kw_times, sem_index_ms, kw_index_ms):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"latency_{timestamp}.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "semantic_ms", "keyword_ms"])
        writer.writerow(["mean_query",   f"{mean(sem_times):.2f}", f"{mean(kw_times):.2f}"])
        writer.writerow(["stdev_query",  f"{stdev(sem_times):.2f}", f"{stdev(kw_times):.2f}"])
        writer.writerow(["min_query",    f"{min(sem_times):.2f}", f"{min(kw_times):.2f}"])
        writer.writerow(["max_query",    f"{max(sem_times):.2f}", f"{max(kw_times):.2f}"])
        writer.writerow(["index_build",  f"{sem_index_ms:.0f}", f"{kw_index_ms:.0f}"])

    print(f"\nResultat sparade: {path}")


if __name__ == "__main__":
    main()

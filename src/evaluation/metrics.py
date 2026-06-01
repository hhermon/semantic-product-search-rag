"""Utvärderingsmått för informationssökning: Precision, Recall, F1 och MAP."""

from dataclasses import dataclass, field


@dataclass
class QueryEval:
    """Resultat för en enskild testfråga."""
    query_id: str
    query: str
    precision: float
    recall: float
    f1: float
    average_precision: float
    group: str = ""


@dataclass
class EvalReport:
    """Aggregerat resultat över alla testfrågor."""
    query_evals: list[QueryEval] = field(default_factory=list)
    mean_average_precision: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Andel relevanta bland de k första resultaten."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Andel av de relevanta som hittats bland de k första resultaten."""
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(relevant_ids)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Average Precision (AP) — area under precision-recall-kurvan."""
    if not relevant_ids:
        return 1.0
    hits = 0
    score = 0.0
    for i, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_ids:
            hits += 1
            score += hits / i
    return score / len(relevant_ids)


def evaluate_query(
    query_id: str,
    query: str,
    results,
    relevant_ids: list[str],
) -> QueryEval:
    retrieved_ids = [r.product.id for r in results]
    k = len(retrieved_ids)
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return QueryEval(
        query_id=query_id,
        query=query,
        precision=p,
        recall=r,
        f1=f1_score(p, r),
        average_precision=average_precision(retrieved_ids, relevant_ids),
    )


def evaluate_all(query_evals: list[QueryEval]) -> EvalReport:
    if not query_evals:
        return EvalReport()
    n = len(query_evals)
    return EvalReport(
        query_evals=query_evals,
        mean_average_precision=sum(e.average_precision for e in query_evals) / n,
        mean_precision=sum(e.precision for e in query_evals) / n,
        mean_recall=sum(e.recall for e in query_evals) / n,
        mean_f1=sum(e.f1 for e in query_evals) / n,
    )

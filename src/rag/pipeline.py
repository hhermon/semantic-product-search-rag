"""RAG-pipeline: hämtar relevanta produkter och genererar svar med Gemini."""

import os
from google import genai
from google.genai import types

from src.retrieval.retriever import ProductRetriever, SearchResult


class RAGPipeline:
    """Retrieval-Augmented Generation för produktsökning.

    Flöde:
    1. Hämta semantiskt relevanta produkter via retriever.
    2. Bygg kontext med produktdetaljer.
    3. Skicka kontext + fråga till Gemini för svarsgenerering.
    """

    SYSTEM_PROMPT = """You are a helpful product assistant for an e-commerce store.
Help customers find the right products based on their needs.

You have access to relevant products from our catalog as context.
Base your recommendations only on the products in the context.
Be concrete, comparative and adapt your answer to the customer's specific needs."""

    def __init__(self, retriever: ProductRetriever, model: str | None = None, top_k: int | None = None):
        self.retriever = retriever
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.top_k = top_k or int(os.getenv("TOP_K", "5"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _build_context(self, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant products found in the catalog."
        parts = ["Relevant products from our catalog:\n"]
        for result in results:
            parts.append(f"[Relevance: {result.score:.2f}]")
            parts.append(result.product.to_detail())
            parts.append("---")
        return "\n".join(parts)

    def query(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
        category_filter: str | None = None,
        max_price: float | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """Kör en RAG-fråga och returnerar svar + hämtade produkter."""
        results = self.retriever.search(
            query=user_query, top_k=self.top_k,
            category_filter=category_filter, max_price=max_price,
        )
        context = self._build_context(results)

        history = []
        for msg in (conversation_history or []):
            role = "user" if msg.get("role") == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=msg.get("content", ""))]))

        chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(system_instruction=self.SYSTEM_PROMPT),
            history=history,
        )
        prompt = f"Product catalog context:\n{context}\n\nCustomer question: {user_query}"
        response = chat.send_message(prompt)
        return response.text, results

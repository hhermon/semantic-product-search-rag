"""AI-agent med planering, verktygsanvändning och flerstegsresonemang."""

import os
from google import genai
from google.genai import types

from src.retrieval.retriever import ProductRetriever

SYSTEM_PROMPT = """You are an intelligent product assistant for an e-commerce store.
Help customers find and explore products by:
1. Understanding the customer's needs, even when vague or complex
2. Planning how to search for relevant products
3. Actively using search tools to retrieve information from the catalog
4. Refining your search if initial results are insufficient
5. Comparing and recommending products based on the customer's situation

Always use tools to search the actual product catalog before giving recommendations.
Feel free to ask follow-up questions if you need more information."""


class ProductAgent:
    """Agentbaserat system med verktygsanvändning och flerstegsresonemang."""

    def __init__(self, retriever: ProductRetriever, model: str | None = None):
        self.retriever = retriever
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _search_products(self, query: str, top_k: int = 5, category: str = "", max_price: float = 0) -> str:
        """Search the product catalog using semantic search.

        Args:
            query: Search query describing what the user is looking for.
            top_k: Number of products to return.
            category: Filter by product category.
            max_price: Maximum price in USD. Set to 0 for no price limit.
        """
        results = self.retriever.search(
            query=query,
            top_k=top_k,
            category_filter=category or None,
            max_price=max_price if max_price > 0 else None,
        )
        if not results:
            return "No products found for this search."
        output = [f"Found {len(results)} product(s):\n"]
        for r in results:
            output.append(f"[Score: {r.score:.2f}] {r.product.to_detail()}\n---")
        return "\n".join(output)

    def _get_product_details(self, product_id: str) -> str:
        """Get detailed information about a specific product.

        Args:
            product_id: Product ID, e.g. 1, 42.
        """
        product = self.retriever._product_map.get(product_id)
        if product is None:
            return f"Product with ID '{product_id}' not found."
        return product.to_detail()

    def _compare_products(self, product_ids: str, attributes: str = "") -> str:
        """Compare multiple products side by side.

        Args:
            product_ids: Comma-separated product IDs, e.g. 1,2,3.
            attributes: Comma-separated attributes to compare, e.g. price,brand.
        """
        ids = [pid.strip() for pid in product_ids.split(",")]
        products = [self.retriever._product_map.get(pid) for pid in ids]
        products = [p for p in products if p is not None]
        if not products:
            return "None of the specified products were found."
        attr_list = [a.strip() for a in attributes.split(",")] if attributes else ["price", "brand", "category"]
        lines = [f"Comparison of {len(products)} products:\n"]
        for attr in attr_list:
            row = [attr] + [str(getattr(p, attr, p.attributes.get(attr, "—"))) for p in products]
            lines.append(" | ".join(row))
        return "\n".join(lines)

    def chat(self, user_message: str, conversation_history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Kör agenten med ett användarmeddelande."""
        history = []
        for msg in (conversation_history or []):
            role = "user" if msg.get("role") == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=msg.get("content", ""))]))

        chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[self._search_products, self._get_product_details, self._compare_products],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            ),
            history=history,
        )

        response = chat.send_message(user_message)
        new_turns = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.text},
        ]
        return response.text, new_turns

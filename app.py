"""Streamlit-app för semantisk produktsökning med RAG och agent-läge."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.retriever import ProductRetriever
from src.retrieval.keyword_search import KeywordSearcher
from src.data.product_loader import ProductLoader

st.set_page_config(
    page_title="Semantisk Produktsökning",
    page_icon="🔍",
    layout="wide",
)

HAS_GEMINI = bool(os.getenv("GEMINI_API_KEY"))


@st.cache_resource(show_spinner="Laddar produktkatalog och bygger sökindex...")
def load_systems():
    loader = ProductLoader()
    products = loader.load()
    retriever = ProductRetriever.from_data_file()
    keyword_searcher = KeywordSearcher.from_products(products)
    return retriever, keyword_searcher, products


@st.cache_resource(show_spinner="Initierar RAG-pipeline...")
def load_rag(_retriever):
    from src.rag.pipeline import RAGPipeline
    return RAGPipeline(retriever=_retriever)


@st.cache_resource(show_spinner="Initierar agent...")
def load_agent(_retriever):
    from src.agent.agent import ProductAgent
    return ProductAgent(retriever=_retriever)


def main():
    retriever, keyword_searcher, products = load_systems()

    with st.sidebar:
        st.title("⚙️ Inställningar")

        mode_options = ["Semantisk", "Nyckelord (TF-IDF)", "Jämför båda"]
        if HAS_GEMINI:
            mode_options += ["RAG (Gemini)", "Agent (Gemini)"]

        search_mode = st.radio(
            "Sökmetod",
            options=mode_options,
            help=(
                "**Semantisk**: Sentence-BERT embeddings + cosinuslikhet\n\n"
                "**Nyckelord**: Klassisk TF-IDF-sökning\n\n"
                "**Jämför**: Visa båda sida vid sida\n\n"
                "**RAG**: Retrieval-Augmented Generation med Gemini\n\n"
                "**Agent**: AI-agent med verktygsanvändning (Gemini)"
            ),
        )

        st.divider()

        if search_mode != "Agent (Gemini)":
            category_filter = st.selectbox(
                "Filtrera kategori",
                options=["Alla"] + sorted({p.category for p in products}),
            )
            category = None if category_filter == "Alla" else category_filter

            top_k = st.slider("Antal resultat", min_value=1, max_value=10, value=5)

            max_price = st.number_input(
                "Maxpris (USD)", min_value=0, max_value=10_000, value=0, step=50,
                help="Sätt till 0 för inget pristak.",
            )
            max_price_filter = float(max_price) if max_price > 0 else None
        else:
            category = None
            top_k = 5
            max_price_filter = None

        st.divider()
        st.markdown("**Om systemet**")
        status = "✅ Konfigurerad" if HAS_GEMINI else "❌ Saknar GEMINI_API_KEY"
        st.caption(
            f"{len(products)} produkter i katalogen\n\n"
            f"Gemini: {status}"
        )

    st.title("🔍 Semantisk Produktsökning")
    st.caption("Prototyp för semantisk produktutforskning i e-handel — RAG & Agent")

    if search_mode == "Agent (Gemini)":
        _agent_chat_ui(retriever)

    elif search_mode == "RAG (Gemini)":
        _rag_chat_ui(retriever, category, max_price_filter)

    else:
        query = st.text_input(
            "Sökfråga",
            placeholder="t.ex. laptop for programming",
        )

        if query:
            if search_mode == "Semantisk":
                _show_semantic(retriever, query, top_k, category, max_price_filter)
            elif search_mode == "Nyckelord (TF-IDF)":
                _show_keyword(keyword_searcher, query, top_k, category, max_price_filter)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Semantisk sökning")
                    _show_semantic(retriever, query, top_k, category, max_price_filter)
                with col2:
                    st.subheader("Nyckelord (TF-IDF)")
                    _show_keyword(keyword_searcher, query, top_k, category, max_price_filter)


def _rag_chat_ui(retriever, category, max_price):
    st.subheader("💬 RAG-konversation")
    st.caption("Ställ en fråga — systemet hämtar relevanta produkter och genererar ett svar med Gemini.")

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    if "rag_results" not in st.session_state:
        st.session_state.rag_results = []

    for msg in st.session_state.rag_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    if st.session_state.rag_results:
        with st.expander(f"📦 Hämtade produkter ({len(st.session_state.rag_results)} st)", expanded=False):
            for r in st.session_state.rag_results:
                st.markdown(f"**{r.product.name}** — {r.product.price} USD (score: {r.score:.2f})")

    user_input = st.chat_input("Skriv din fråga här...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Hämtar produkter och genererar svar..."):
                try:
                    pipeline = load_rag(retriever)
                    answer, results = pipeline.query(
                        user_query=user_input,
                        conversation_history=st.session_state.rag_history,
                        category_filter=category,
                        max_price=max_price,
                    )
                    st.write(answer)
                    st.session_state.rag_results = results
                except Exception as e:
                    answer = f"Fel: {e}"
                    st.error(answer)
                    st.session_state.rag_results = []

        st.session_state.rag_history.append({"role": "user", "content": user_input})
        st.session_state.rag_history.append({"role": "assistant", "content": answer})

    if st.session_state.rag_history:
        if st.button("🗑️ Rensa konversation", key="clear_rag"):
            st.session_state.rag_history = []
            st.session_state.rag_results = []
            st.rerun()


def _agent_chat_ui(retriever):
    st.subheader("🤖 AI-Agent")
    st.caption("En intelligent agent som planerar och söker i produktkatalogen med verktygsanvändning.")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for msg in st.session_state.agent_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])

    user_input = st.chat_input("Beskriv vad du letar efter...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Agenten arbetar..."):
                try:
                    agent = load_agent(retriever)
                    answer, new_turns = agent.chat(
                        user_message=user_input,
                        conversation_history=st.session_state.agent_history,
                    )
                    st.write(answer)
                except Exception as e:
                    answer = f"Fel: {e}"
                    new_turns = [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": answer},
                    ]
                    st.error(answer)

        st.session_state.agent_history.extend(new_turns)

    if st.session_state.agent_history:
        if st.button("🗑️ Rensa konversation", key="clear_agent"):
            st.session_state.agent_history = []
            st.rerun()


def _show_semantic(retriever, query, top_k, category, max_price):
    results = retriever.search(query=query, top_k=top_k, category_filter=category, max_price=max_price)
    if not results:
        st.info("Inga produkter hittades.")
        return
    for r in results:
        _render_product_card(r.product, r.score, r.rank, color="#1f77b4")


def _show_keyword(searcher, query, top_k, category, max_price):
    results = searcher.search(query=query, top_k=top_k, category_filter=category, max_price=max_price)
    if not results:
        st.info("Inga produkter hittades.")
        return
    for r in results:
        _render_product_card(r.product, r.score, r.rank, color="#ff7f0e")


def _render_product_card(product, score, rank, color):
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**#{rank} {product.name}**")
            st.caption(f"{product.brand} · {product.category}")
            st.write(product.description[:180] + "...")
            tags = " ".join(f"`{t}`" for t in product.tags[:4])
            st.markdown(tags)
        with col2:
            st.metric("Pris", f"{product.price:,.0f} USD")
            st.markdown(
                f"<div style='color:{color}; font-weight:bold; font-size:0.9em'>Score: {score:.3f}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

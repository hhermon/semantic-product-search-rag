"""Streamlit-app för semantisk produktsökning — Vecka 1: Retrieval."""

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


@st.cache_resource(show_spinner="Laddar produktkatalog och bygger sökindex...")
def load_systems():
    loader = ProductLoader()
    products = loader.load()
    retriever = ProductRetriever.from_data_file()
    keyword_searcher = KeywordSearcher.from_products(products)
    return retriever, keyword_searcher, products


def main():
    retriever, keyword_searcher, products = load_systems()

    with st.sidebar:
        st.title("⚙️ Inställningar")

        search_mode = st.radio(
            "Sökmetod",
            options=["Semantisk", "Nyckelord (TF-IDF)", "Jämför båda"],
            help=(
                "**Semantisk**: Sentence-BERT embeddings + cosinuslikhet\n\n"
                "**Nyckelord**: Klassisk TF-IDF-sökning\n\n"
                "**Jämför**: Visa båda sida vid sida"
            ),
        )

        st.divider()

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

        st.divider()
        st.markdown("**Om systemet**")
        st.caption(
            f"{len(products)} produkter i katalogen\n\n"
            "Semantisk sökning med Sentence-BERT\n"
            "Nyckelordssökning med TF-IDF"
        )

    st.title("🔍 Semantisk Produktsökning")
    st.caption("Prototyp för semantisk produktutforskning i e-handel")

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

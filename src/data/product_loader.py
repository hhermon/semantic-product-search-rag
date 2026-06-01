"""Laddning och modellering av produktdata från Amazon Products Dataset 2023."""

from pathlib import Path
from pydantic import BaseModel, Field


class Product(BaseModel):
    """Produktmodell med både strukturerad och ostrukturerad data."""

    id: str
    name: str
    category: str
    price: float
    currency: str = "USD"
    brand: str
    attributes: dict = Field(default_factory=dict)
    description: str
    tags: list[str] = Field(default_factory=list)

    def to_search_text(self) -> str:
        attr_text = " ".join(f"{k}: {v}" for k, v in self.attributes.items())
        tags_text = " ".join(self.tags)
        return f"{self.name} {self.category} {self.brand} {self.description} {attr_text} {tags_text}"

    def to_summary(self) -> str:
        return (
            f"**{self.name}** ({self.brand})\n"
            f"Kategori: {self.category} | Pris: {self.price} {self.currency}\n"
            f"{self.description[:150]}..."
        )

    def to_detail(self) -> str:
        attrs = "\n".join(f"  - {k}: {v}" for k, v in self.attributes.items())
        tags = ", ".join(self.tags)
        return (
            f"Produkt: {self.name}\n"
            f"ID: {self.id}\n"
            f"Varumärke: {self.brand}\n"
            f"Kategori: {self.category}\n"
            f"Pris: {self.price} {self.currency}\n"
            f"Attribut:\n{attrs}\n"
            f"Beskrivning: {self.description}\n"
            f"Taggar: {tags}"
        )


# Kategori-ID:n och antal produkter per kategori (stratifierat urval)
_SAMPLE_CATEGORIES = {
    113: 500,  # Men's Watches
    114: 500,  # Men's Shoes
    122: 500,  # Women's Shoes
     71: 500,  # Headphones & Earbuds
    107: 500,  # Backpacks
     49: 500,  # Skin Care Products
    270: 500,  # Toys & Games
    110: 500,  # Men's Clothing
    108: 500,  # Luggage
    170: 500,  # Kitchen & Dining
}


def _load_amazon(seed: int = 42) -> list[Product]:
    """Laddar stratifierat urval från Amazon Products Dataset (lokal CSV)."""
    import pandas as pd

    data_dir = Path(__file__).parent
    df = pd.read_csv(data_dir / "amazon_products.csv", low_memory=False)
    cats = pd.read_csv(data_dir / "amazon_categories.csv")
    cat_map = dict(zip(cats["id"], cats["category_name"]))

    df = df.dropna(subset=["title", "price"])
    df = df[df["price"] > 0]
    df = df[df["title"].str.len() > 5]

    frames = []
    for cat_id, n in _SAMPLE_CATEGORIES.items():
        subset = df[df["category_id"] == cat_id]
        frames.append(subset.sample(n=min(n, len(subset)), random_state=seed))

    sampled = pd.concat(frames).reset_index(drop=True)

    products = []
    for row in sampled.itertuples(index=False):
        category = cat_map.get(row.category_id, "unknown")
        attributes = {}
        stars = getattr(row, "stars", None)
        reviews = getattr(row, "reviews", None)
        bought = getattr(row, "boughtInLastMonth", None)
        import math
        if stars and not (isinstance(stars, float) and math.isnan(stars)):
            attributes["rating"] = round(float(stars), 1)
        if reviews and not (isinstance(reviews, float) and math.isnan(reviews)):
            attributes["reviews"] = int(reviews)
        if bought and not (isinstance(bought, float) and math.isnan(bought)):
            attributes["bought_last_month"] = int(bought)

        title = str(row.title)[:200]
        products.append(Product(
            id=str(row.asin),
            name=title,
            category=category,
            price=float(row.price),
            currency="USD",
            brand=category,
            attributes=attributes,
            description=title,
            tags=[category],
        ))

    return products


class ProductLoader:
    """Laddar produkter från Amazon Products Dataset 2023 (Kaggle, ODC-By licens).

    Använder stratifierat urval: 500 produkter per kategori, totalt 5 000 produkter
    fördelade över 10 kategorier (klockor, skor, hörlurar, ryggsäckar m.fl.).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def load(self) -> list[Product]:
        return _load_amazon(self.seed)

    def load_by_category(self, category: str) -> list[Product]:
        return [p for p in self.load() if p.category.lower() == category.lower()]

    def load_by_price_range(self, min_price: float, max_price: float) -> list[Product]:
        return [p for p in self.load() if min_price <= p.price <= max_price]

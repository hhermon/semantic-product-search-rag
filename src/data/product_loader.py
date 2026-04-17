"""Laddning och modellering av produktdata från DummyJSON API."""

import json
import urllib.request
from pathlib import Path
from typing import Optional
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
        """Kombinerar namn, beskrivning och taggar till en sökbar textsträng."""
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


def _fetch_dummyjson(limit: int = 194) -> list[Product]:
    """Hämtar produkter från DummyJSON API och konverterar till Product-modell."""
    url = f"https://dummyjson.com/products?limit={limit}&select=id,title,description,price,brand,category,tags,dimensions,weight,rating,stock"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    products = []
    for item in data.get("products", []):
        attributes = {}
        if item.get("weight"):
            attributes["weight"] = item["weight"]
        if item.get("dimensions"):
            d = item["dimensions"]
            attributes["dimensions"] = f"{d.get('width')}x{d.get('height')}x{d.get('depth')} cm"
        if item.get("rating"):
            attributes["rating"] = item["rating"]
        if item.get("stock"):
            attributes["stock"] = item["stock"]

        products.append(Product(
            id=str(item["id"]),
            name=item.get("title", ""),
            category=item.get("category", ""),
            price=float(item.get("price", 0)),
            currency="USD",
            brand=item.get("brand") or item.get("category", ""),
            attributes=attributes,
            description=item.get("description", ""),
            tags=item.get("tags") or [],
        ))
    return products


class ProductLoader:
    """Laddar produkter från DummyJSON API."""

    def __init__(self, data_path: Optional[str] = None, limit: int = 194):
        self.limit = limit

    def load(self) -> list[Product]:
        """Hämtar och returnerar alla produkter från API:et."""
        return _fetch_dummyjson(self.limit)

    def load_by_category(self, category: str) -> list[Product]:
        return [p for p in self.load() if p.category.lower() == category.lower()]

    def load_by_price_range(self, min_price: float, max_price: float) -> list[Product]:
        return [p for p in self.load() if min_price <= p.price <= max_price]

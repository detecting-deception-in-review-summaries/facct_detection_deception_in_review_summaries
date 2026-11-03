from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .common import digest, stable_json
from .models import Product, Review


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def review_field(review: Review | dict, field: str):
    if isinstance(review, Review):
        return getattr(review, field)

    return review.get(field)


def format_reviews(reviews: Iterable[Review | dict]) -> str:
    blocks: list[str] = []

    for number, review in enumerate(reviews, 1):
        title = clean(str(review_field(review, "title") or ""))
        body = clean(str(review_field(review, "text") or ""))

        lines = [f"Review {number}:"]

        if title:
            lines.append(title)

        lines.append(body)
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def shuffle_reviews(
    reviews: list[Review],
    product_id: str,
    order_seed: int,
) -> list[Review]:
    """Return one product's reviews in a reproducible pseudo-random order.

    Presentation order is a confound: a fixed order leaves the same reviews in
    the same prompt positions for every condition and model, so any position
    effect is systematic rather than noise that averages out. Shuffling spreads
    it across the corpus while a seed keeps the run reproducible.

    The seed is derived per product rather than drawn from one shared stream,
    so a product's order does not depend on how many products were loaded
    alongside it. That is what keeps a four-product `--dry-run` consistent with
    the full run, and keeps `source_hash` a function of the product alone.
    """

    derived_seed = int(digest(stable_json([order_seed, product_id]), 16), 16)
    ordered = list(reviews)
    random.Random(derived_seed).shuffle(ordered)

    return ordered


def load_products(
    path: str | Path,
    dry_run: bool = False,
    order_seed: int | None = None,
) -> list[Product]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    by_product: dict[str, list[dict]] = defaultdict(list)

    for review in raw["reviews"]:
        by_product[str(review["product_id"])].append(review)

    metas = raw["products"]

    if dry_run:
        selected: list[dict] = []

        for source in ("amazon", "google_maps", "yelp", "hotel_rec"):
            source_products = sorted(
                (product for product in metas if product["source"] == source),
                key=lambda product: product["product_id"],
            )

            selected.append(source_products[0])

        metas = selected

    products: list[Product] = []

    for meta in metas:
        # Selecting the reviews stays chronological and deterministic, so which
        # 100 a product contributes never depends on the ordering seed; only
        # the order they are presented in does.
        source_reviews = sorted(
            by_product[str(meta["product_id"])],
            key=lambda review: (
                str(review.get("timestamp", "")),
                str(review.get("reviewer_id", "")),
                review["text"],
            ),
        )[:100]

        reviews = [
            Review(
                product_id=str(meta["product_id"]),
                title=review.get("title"),
                text=clean(review["text"]),
                rating=review.get("rating"),
            )
            for review in source_reviews
        ]

        if order_seed is not None:
            reviews = shuffle_reviews(
                reviews,
                product_id=str(meta["product_id"]),
                order_seed=order_seed,
            )

        source_text = format_reviews(reviews)
        dataset_role = (
            "threshold_calibration"
            if meta["source"] == "hotel_rec"
            else "main_evaluation"
        )

        products.append(
            Product(
                product_id=str(meta["product_id"]),
                source=meta["source"],
                category=meta["category"],
                review_count_bin=meta["range"],
                dataset_role=dataset_role,
                reviews=reviews,
                source_text=source_text,
                source_hash=digest(source_text),
            )
        )

    main_product_ids = {
        product.product_id
        for product in products
        if product.dataset_role == "main_evaluation"
    }
    calibration_product_ids = {
        product.product_id
        for product in products
        if product.dataset_role == "threshold_calibration"
    }

    if main_product_ids & calibration_product_ids:
        raise ValueError("A product appears in both dataset roles.")

    return products

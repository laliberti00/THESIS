"""Foursquare legacy v2 category taxonomy → top-level macro lookup.

The taxonomy JSON is committed in `config/foursquare_legacy_taxonomy.json` and is
read from disk; there is NO runtime API call to Foursquare. The file mirror used
is `clinejj/foursquare-api-java` (the categories_1.json test resource).

This module exposes one function:

    build_cat_id_to_macro(taxonomy_path) -> dict[str, str]

It walks the tree and returns a mapping ``cat_id (hex) → top-level macro name``.
Top-level categories themselves have no `id` field in the v2 dump; only their
children (and descendants) do, so we map every descendant of a top-level node to
that top-level name. Coverage on TSMC2014 (NYC ∪ TKY) is ~80 %; the remaining
~20 % (categories added to the taxonomy after 2014) are intentionally mapped to
``"Other"`` by the caller — see brief.

Top-level nodes found in this dump (8): Arts & Entertainment, College &
University, Food, Great Outdoors, Home/Work/Other, Nightlife Spot, Shop, Travel
Spot.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_cat_id_to_macro(taxonomy_path: Path | str) -> dict[str, str]:
    """Parse the Foursquare v2 legacy taxonomy JSON and return a flat mapping
    ``cat_id → top_level_macro_name``.

    Args:
        taxonomy_path: path to ``config/foursquare_legacy_taxonomy.json``.

    Returns:
        dict where keys are the hex ``cat_id`` strings and values are one of
        the 8 v2 top-level names.

    Notes:
        - File is read with ``latin-1`` (the JSON ships with non-UTF8 bytes,
          presumably in category names).
        - Top-level nodes do NOT have an ``id`` field in this dump, so they are
          not entered into the map under themselves; only their descendants are.
    """
    path = Path(taxonomy_path)
    data = json.loads(path.read_text(encoding="latin-1"))
    top = data["response"]["categories"]

    cat2macro: dict[str, str] = {}

    def walk(nodes, top_name: str) -> None:
        for n in nodes:
            if "id" in n:
                cat2macro[n["id"]] = top_name
            for child in n.get("categories", []):
                walk([child], top_name)

    for c in top:
        walk(c.get("categories", []), c["name"])
    return cat2macro

"""Foursquare legacy v2 category taxonomy → top-level macro lookup.

The taxonomy JSON is committed in `config/foursquare_legacy_taxonomy.json` and is
read from disk; there is NO runtime API call to Foursquare. The current mirror
is the gist `a0344050d3e2b52256d7/foursquare-taxonomy.json` (10 top-level
categories, 606 IDs, top-levels carry their own id) — a richer dump than the
older `clinejj/foursquare-api-java/categories_1.json` (8 top-level, 344 IDs,
top-levels missing their id).

This module exposes one function:

    build_cat_id_to_macro(taxonomy_path) -> dict[str, str]

It walks the tree and returns a mapping ``cat_id (hex) → top-level macro name``.
The function transparently handles both schemas seen in the wild:

    SchemaA (gist 2015):  top-level is a list of {"id","name","categories":[...]}
    SchemaB (clinejj):    top-level is {"response":{"categories":[{"name","categories":[...]}, ...]}}

Coverage on TSMC2014 (NYC ∪ TKY): 99.5 % (400 / 402) with SchemaA, ~81 % with
SchemaB. Uncovered IDs are intentionally mapped to ``"Other"`` by the caller.

Top-level nodes from SchemaA (the 10 canonical v2 macros): Arts & Entertainment,
College & University, Event, Food, Nightlife Spot, Outdoors & Recreation,
Professional & Other Places, Residence, Shop & Service, Travel & Transport.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_cat_id_to_macro(taxonomy_path: Path | str) -> dict[str, str]:
    """Parse the Foursquare v2 legacy taxonomy JSON and return a flat mapping
    ``cat_id → top_level_macro_name``.

    Accepts both the SchemaA (flat list at top, top-levels carry id) and the
    SchemaB (response.categories envelope, top-levels missing id) variants.
    """
    path = Path(taxonomy_path)
    data = json.loads(path.read_text(encoding="latin-1"))

    # Schema detection
    if isinstance(data, list):
        top = data                                            # SchemaA
    elif isinstance(data, dict) and "response" in data:
        top = data["response"]["categories"]                  # SchemaB
    else:
        raise ValueError(f"Unrecognised taxonomy schema at {path}")

    cat2macro: dict[str, str] = {}

    def walk(nodes, top_name: str) -> None:
        for n in nodes:
            if "id" in n:
                cat2macro[n["id"]] = top_name
            for child in n.get("categories", []):
                walk([child], top_name)

    for c in top:
        # SchemaA top-levels carry their own id — register it under itself
        if "id" in c:
            cat2macro[c["id"]] = c["name"]
        walk(c.get("categories", []), c["name"])
    return cat2macro

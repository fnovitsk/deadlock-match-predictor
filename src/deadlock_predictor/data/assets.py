"""Hero and item names, from assets.deadlock-api.com.

The names are fetched once and cached in `data/assets/` (heroes.json, items.json). That
host doesn't resolve on every network; if it fails, save the two JSON files there by hand
from a browser that can reach it:
- https://assets.deadlock-api.com/v2/heroes → data/assets/heroes.json
- https://assets.deadlock-api.com/v2/items → data/assets/items.json
Without them, names fall back to ids ("Hero 12", "Item 267712790").
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import requests

from deadlock_predictor.data.dumps import USER_AGENT
from deadlock_predictor.data.match import ROOT

ASSETS_DIR = ROOT / "data" / "assets"
ASSETS_URL = "https://assets.deadlock-api.com/v2/{kind}"


def load(kind: str, assets_dir: Path = ASSETS_DIR, timeout: float = 10) -> list[dict]:
    """The asset list for `kind` ('heroes' or 'items'): cached, else fetched, else []."""
    path = assets_dir / f"{kind}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        response = requests.get(
            ASSETS_URL.format(kind=kind), headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []
    assets_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def names(entries: list[dict]) -> dict[int, str]:
    """{id: name} from an asset list."""
    return {e["id"]: e["name"] for e in entries if "id" in e and e.get("name")}


@cache
def hero_names() -> dict[int, str]:
    return names(load("heroes"))


@cache
def item_names() -> dict[int, str]:
    return names(load("items"))


def hero_name(hero_id: int) -> str:
    return hero_names().get(hero_id, f"Hero {hero_id}")


def item_name(item_id: int) -> str:
    return item_names().get(item_id, f"Item {item_id}")

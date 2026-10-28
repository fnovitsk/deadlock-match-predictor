"""Hero and item names, from the API's static assets (`/v1/assets/heroes`, `/items`).

The names are fetched once and cached in `data/assets/` (heroes.json, items.json); delete
those files to refresh them after a game update. Without network, names fall back to ids
("Hero 12", "Item 267712790"). The assets used to live on assets.deadlock-api.com, which no
longer exists.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import requests

from deadlock_predictor.data.dumps import USER_AGENT
from deadlock_predictor.data.match import ROOT

ASSETS_DIR = ROOT / "data" / "assets"
ASSETS_URL = "https://api.deadlock-api.com/v1/assets/{kind}"


def load(kind: str, assets_dir: Path = ASSETS_DIR, timeout: float = 60) -> list[dict]:
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

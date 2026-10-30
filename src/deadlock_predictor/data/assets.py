"""Hero and rank assets, from the API's static assets (`/v1/assets/...`).

The lists are fetched once and cached in `data/assets/` (heroes.json, ranks.json), and
images in `data/assets/images/`; delete them to refresh after a game update. Without
network, names fall back to ids ("Hero 12") and images to None. The assets used to live
on assets.deadlock-api.com, which no longer exists.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import requests

from deadlock_predictor.data.dumps import USER_AGENT
from deadlock_predictor.data.match import ROOT

ASSETS_DIR = ROOT / "data" / "assets"
ASSETS_URL = "https://api.deadlock-api.com/v1/assets/{kind}"
IMAGES_DIR = ASSETS_DIR / "images"
MIME = {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


def load(kind: str, assets_dir: Path = ASSETS_DIR, timeout: float = 60) -> list[dict] | dict:
    """The asset data for `kind` ('heroes', 'ranks'): cached, else fetched, else []."""
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


def hero_name(hero_id: int) -> str:
    return hero_names().get(hero_id, f"Hero {hero_id}")


def image(url: str | None, images_dir: Path = IMAGES_DIR, timeout: float = 30) -> str | None:
    """A data URI for the image at `url`, downloaded once into `images_dir`; None if it
    can't be had. Data URIs keep the page working without the asset host once cached."""
    if not url:
        return None
    suffix = Path(url.split("?")[0]).suffix.lower() or ".png"
    path = images_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + suffix)
    if not path.exists():
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None
        images_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{MIME.get(suffix, 'image/png')};base64,{data}"


@dataclass(frozen=True)
class Hero:
    id: int
    name: str
    color: str  # the hero's own colour, "#RRGGBB"
    portrait: str | None  # data URI (or None)


@cache
def _heroes_by_id() -> dict[int, dict]:
    return {h["id"]: h for h in load("heroes") if "id" in h}


@cache
def hero(hero_id: int) -> Hero:
    h = _heroes_by_id().get(hero_id, {})
    images = h.get("images") or {}
    # `style_hex` is distinct for every hero; `ui` is shared by several.
    color = (h.get("colors") or {}).get("style_hex") or "#969696"
    return Hero(
        id=hero_id,
        name=h.get("name") or f"Hero {hero_id}",
        color=color.lower(),
        portrait=image(images.get("icon_image_small")),
    )


@cache
def rank(badge: int) -> tuple[str, str | None]:
    """The rank name ("Acolyte 3") and badge image of a badge (tier·10 + subtier)."""
    tier, sub = divmod(int(badge), 10)
    entry = next((r for r in load("ranks") if r.get("tier") == tier), {})
    name = entry.get("name") or f"Tier {tier}"
    images = entry.get("images") or {}
    return (f"{name} {sub}" if sub else name), image(images.get(f"large_subrank{sub}"))

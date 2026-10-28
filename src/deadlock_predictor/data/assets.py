"""Hero, item and map assets, from the API's static assets (`/v1/assets/...`).

The lists are fetched once and cached in `data/assets/` (heroes.json, items.json,
map.json), and images in `data/assets/images/`; delete them to refresh after a game
update. Without network, names fall back to ids ("Hero 12", "Item 267712790") and images
to None. The assets used to live on assets.deadlock-api.com, which no longer exists.
"""

from __future__ import annotations

import base64
import hashlib
import io
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
    """The asset data for `kind` ('heroes', 'items', 'map'): cached, else fetched, else []."""
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
    portrait: str | None  # data URIs (or None)
    card: str | None
    minimap: str | None


@dataclass(frozen=True)
class Item:
    id: int
    name: str
    tier: int  # 1-5 (0 if not a shop item)
    cost: int
    slot: str  # weapon, vitality, spirit ("" if not a shop item)
    icon: str | None
    shop: bool  # a shop upgrade (not an ability unlock or other entry)


@cache
def _heroes_by_id() -> dict[int, dict]:
    return {h["id"]: h for h in load("heroes") if "id" in h}


@cache
def _items_by_id() -> dict[int, dict]:
    return {i["id"]: i for i in load("items") if "id" in i}


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
        card=image(images.get("icon_hero_card")),
        minimap=image(images.get("minimap_image")),
    )


@cache
def item(item_id: int) -> Item:
    i = _items_by_id().get(item_id, {})
    shop = i.get("type") == "upgrade"
    return Item(
        id=item_id,
        name=i.get("name") or f"Item {item_id}",
        tier=int(i.get("item_tier") or 0) if shop else 0,
        cost=int(i.get("cost") or 0) if shop else 0,
        slot=i.get("item_slot_type") or "" if shop else "",
        icon=image(i.get("image")) if shop else None,
        shop=shop,
    )


@dataclass(frozen=True)
class MapInfo:
    radius: float  # world units from the centre to the image's edge; y points up
    image: str | None  # data URI of the minimap
    dark: str | None  # the same with the streets recoloured for a dark page


def tinted(data_uri: str | None, color: tuple[int, int, int]) -> str | None:
    """The image with every visible pixel painted `color`, keeping its transparency (and
    the shading, through the alpha channel). Turns the grey minimap into one that reads on
    a dark page."""
    if not data_uri:
        return None
    from PIL import Image

    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    grey = img.convert("L")
    # Darker source pixels (streets) get the full colour, lighter ones (edges) a lighter tint.
    out = Image.merge(
        "RGBA",
        [grey.point(lambda v, c=c: min(255, c + v // 4)) for c in color] + [img.getchannel("A")],
    )
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@cache
def map_info() -> MapInfo:
    """The minimap image and its scale. The asset's own building positions
    (`objective_positions`) don't line up with the map, so buildings come from
    `configs/buildings.yaml` instead (see `viz.map_replay.estimate_buildings`)."""
    data = load("map") or {}
    images = data.get("images") or {}
    minimap = image(images.get("minimap"))
    return MapInfo(
        radius=float(data.get("radius") or 10752), image=minimap, dark=tinted(minimap, (52, 60, 76))
    )

import json

import requests

from deadlock_predictor.data import assets


def test_load_reads_the_cache(tmp_path):
    (tmp_path / "heroes.json").write_text(json.dumps([{"id": 1, "name": "Infernus"}]))
    assert assets.names(assets.load("heroes", tmp_path)) == {1: "Infernus"}


def test_load_without_network_gives_nothing(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(assets.requests, "get", offline)
    assert assets.load("items", tmp_path) == []
    assert not (tmp_path / "items.json").exists()


def test_names_fall_back_to_ids(monkeypatch):
    monkeypatch.setattr(assets, "hero_names", lambda: {1: "Infernus"})
    assert assets.hero_name(1) == "Infernus"
    assert assets.hero_name(99) == "Hero 99"


class FakeResponse:
    content = b"\x89PNG fake"

    def raise_for_status(self):
        pass


def test_image_is_downloaded_once(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(assets.requests, "get", get)
    first = assets.image("https://x/a.png", tmp_path)
    second = assets.image("https://x/a.png", tmp_path)
    assert first == second and first.startswith("data:image/png;base64,")
    assert len(calls) == 1 and len(list(tmp_path.iterdir())) == 1


def test_image_without_network(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(assets.requests, "get", offline)
    assert assets.image("https://x/a.png", tmp_path) is None
    assert assets.image(None, tmp_path) is None


def test_hero_and_item_details(monkeypatch):
    heroes = {
        7: {
            "id": 7,
            "name": "Wraith",
            "colors": {"style_hex": "#FF0010", "ui": [1, 2, 3]},
            "images": {},
        }
    }
    items = {
        5: {
            "id": 5,
            "name": "Extended Magazine",
            "type": "upgrade",
            "item_tier": 1,
            "cost": 800,
            "item_slot_type": "weapon",
        },
        6: {"id": 6, "name": "Some Ability", "type": "ability"},
    }
    monkeypatch.setattr(assets, "_heroes_by_id", lambda: heroes)
    monkeypatch.setattr(assets, "_items_by_id", lambda: items)
    monkeypatch.setattr(assets, "image", lambda url: None)
    for f in (assets.hero, assets.item):
        f.cache_clear()
    h = assets.hero(7)
    assert (h.name, h.color) == ("Wraith", "#ff0010")
    assert assets.hero(99).name == "Hero 99"
    magazine, ability = assets.item(5), assets.item(6)
    assert (magazine.tier, magazine.cost, magazine.slot, magazine.shop) == (1, 800, "weapon", True)
    assert not ability.shop and ability.tier == 0
    for f in (assets.hero, assets.item):
        f.cache_clear()

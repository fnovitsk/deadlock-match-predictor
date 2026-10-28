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

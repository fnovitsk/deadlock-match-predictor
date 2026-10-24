from deadlock_predictor.data.dumps import match_player_urls

MANIFEST = {
    "public_url": "https://example.test/",
    "tables": {
        "match_player": {
            "files": [
                {"kind": "base", "partition": 118, "key": "base/part-118.parquet"},
                {"kind": "base", "partition": 119, "key": "base/part-119.parquet"},
                {"kind": "base", "partition": 120, "key": "base/part-120.parquet"},
                {"kind": "delta", "key": "delta/old.parquet", "rows_by_partition": {"61": 5}},
                {
                    "kind": "delta",
                    "key": "delta/new.parquet",
                    "rows_by_partition": {"62": 1, "120": 9},
                },
            ]
        }
    },
}


def test_all_files_without_range():
    urls = match_player_urls(MANIFEST)
    assert len(urls) == 5
    assert urls[0] == "https://example.test/base/part-118.parquet"


def test_partition_range_filters_base_and_delta_files():
    urls = match_player_urls(MANIFEST, min_partition=119, max_partition=120)
    assert urls == [
        "https://example.test/base/part-119.parquet",
        "https://example.test/base/part-120.parquet",
        "https://example.test/delta/new.parquet",
    ]


def test_exclude_deltas():
    urls = match_player_urls(MANIFEST, min_partition=120, include_deltas=False)
    assert urls == ["https://example.test/base/part-120.parquet"]

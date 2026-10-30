"""The dashboard's logic, kept apart from Streamlit so it can be tested."""

from __future__ import annotations

import requests


def api_error_message(match_id: str, error: Exception) -> str:
    """A readable reason the API couldn't give a match.

    deadlock-api serves matches it has already processed at once. Newer ones it fetches from
    Steam, which can be busy (503) and is rate-limited per IP (429).
    """
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429:
        try:
            wait = response.json()["error"]["next_request_in"]
            when = f" Try again in about {max(1, round(wait / 60))} min."
        except (ValueError, KeyError, TypeError):
            when = ""
        return (
            f"Match {match_id} isn't processed by deadlock-api yet, and fetching it from "
            f"Steam is rate-limited (a few per hour).{when}"
        )
    if status == 503:
        return (
            f"Match {match_id} isn't processed by deadlock-api yet, and Steam is busy. "
            "Try again in a few minutes."
        )
    if status == 404:
        return f"No match {match_id} on deadlock-api."
    if isinstance(error, requests.RequestException):
        return f"Couldn't reach deadlock-api for match {match_id}: {error}"
    return f"Couldn't load match {match_id}: {error}"

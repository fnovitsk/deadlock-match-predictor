import requests

from deadlock_predictor.app.helpers import api_error_message


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def json(self):
        return self.body


def http_error(status, body=None):
    return requests.HTTPError(response=FakeResponse(status, body or {}))


def test_api_error_messages():
    limited = http_error(429, {"error": {"next_request_in": 1790}})
    assert "about 30 min" in api_error_message("5", limited)
    assert "Steam is busy" in api_error_message("5", http_error(503))
    assert api_error_message("5", http_error(404)) == "No match 5 on deadlock-api."
    assert "boom" in api_error_message("5", ValueError("boom"))

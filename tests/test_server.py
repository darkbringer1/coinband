import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from coinband.binance_data import Klines
from coinband.server import App, ModelCache, make_handler, parse_request
from tests.test_backtest import FakeForecaster


def fake_klines(symbol, interval, bars, progress=None):
    close = list(100 + np.cumsum(np.sin(np.arange(bars) / 5)))
    t = [i * 3_600_000 for i in range(bars)]
    if progress:
        progress(bars, bars)
    return Klines(symbol, interval, t, close, close, close, close, [1.0] * bars)


@pytest.fixture()
def server():
    app = App(
        ModelCache("cpu", loader=lambda model_id, device: FakeForecaster()),
        klines_fetcher=fake_klines,
        symbols_fetcher=lambda: [{"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT"}],
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_backtest_job_roundtrip(server):
    status, body = post(f"{server}/api/backtest", {"symbol": "btcusdt", "interval": "1h", "bars": 500, "horizon": 12})
    assert status == 202
    job_id = body["job_id"]
    for _ in range(100):
        _, _, raw = get(f"{server}/api/jobs/{job_id}")
        job = json.loads(raw)
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    r = job["result"]
    assert r["symbol"] == "BTCUSDT"
    assert len(r["close"]) == 500 and len(r["time"]) == 500
    assert r["origins"][-1]["index"] == 500
    assert r["interval_ms"] == 3_600_000


def test_models_and_symbols(server):
    _, _, raw = get(f"{server}/api/models")
    models = json.loads(raw)
    assert models["default"] in {m["id"] for m in models["models"]}
    _, _, raw = get(f"{server}/api/symbols")
    assert json.loads(raw)["symbols"][0]["symbol"] == "BTCUSDT"


def test_static_and_traversal(server):
    status, ctype, body = get(f"{server}/")
    assert status == 200 and ctype.startswith("text/html") and b"coinband" in body
    status, ctype, _ = get(f"{server}/app.js")
    assert "javascript" in ctype
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{server}/../server.py")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{server}/%2e%2e/server.py")
    assert e.value.code == 404


def test_bad_request(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{server}/api/backtest", {"symbol": "BTC/USDT"})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{server}/api/jobs/nope")
    assert e.value.code == 404


@pytest.mark.parametrize(
    "body",
    [{"symbol": "BTCUSDT", "interval": "7m"}, {"symbol": "BTCUSDT", "horizon": 0}, {"symbol": "BTCUSDT", "model": "x"}],
)
def test_parse_request_rejects(body):
    with pytest.raises(ValueError):
        parse_request(body)

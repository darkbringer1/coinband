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
    return Klines(symbol, interval, t, close, close, close, close, [1.0] * bars, {"continuous": True})


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


def wait_job(server, job_id):
    for _ in range(200):
        _, _, raw = get(f"{server}/api/jobs/{job_id}")
        job = json.loads(raw)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_leaderboard_job(server):
    status, body = post(
        f"{server}/api/leaderboard",
        {"symbols": "BTCUSDT, ethusdt BTCUSDT", "bars": 400, "horizon": 8, "max_origins": 50, "trend_window": 24},
    )
    assert status == 202
    job = wait_job(server, body["job_id"])
    assert job["status"] == "done", job
    rows = job["result"]["rows"]
    assert [r["symbol"] for r in rows] == ["BTCUSDT", "ETHUSDT"]
    assert rows[0]["summary"]["scored"] > 0
    assert "by_horizon" not in rows[0]["summary"]
    assert rows[0]["plain_crps_skill"] is not None


def test_backtest_result_has_analysis(server):
    _, body = post(f"{server}/api/backtest", {"symbol": "BTCUSDT", "bars": 400, "horizon": 8, "max_origins": 50})
    r = wait_job(server, body["job_id"])["result"]
    assert len(r["summary"]["by_horizon"]["crps_skill"]) == 8
    assert r["params"]["trend_window"] == 0 and r["trend"] is False


def test_offload_moves_nested_tensors():
    import dataclasses

    import torch

    from coinband.device import Offload, _to

    @dataclasses.dataclass
    class Mask:
        mask: torch.Tensor
        enc_len: int

    moved = _to((torch.ones(2), Mask(torch.zeros(1), 3), {"k": [torch.ones(1)]}), "cpu")
    assert moved[1].enc_len == 3 and isinstance(moved[1].mask, torch.Tensor)
    lin = torch.nn.Linear(2, 2)
    off = Offload(lin, "cpu")
    x = torch.ones(1, 2)
    assert torch.allclose(off(x), lin(x))


def test_paper_page(server):
    status, ctype, body = get(f"{server}/paper")
    assert status == 200 and ctype.startswith("text/html")
    assert b"<h1 id=" in body and b"<table>" in body and b"<details>" in body and b"/paper/figures/skill-by-horizon.svg" in body
    status, ctype, _ = get(f"{server}/paper/figures/skill-by-horizon.svg")
    assert status == 200 and "svg" in ctype
    with pytest.raises(urllib.error.HTTPError):
        get(f"{server}/paper/figures/..%2f..%2fREADME.md")


def test_live_endpoint_returns_new_bars_and_forecast(server):
    # the fake source returns context + 8 = 136 hourly bars, the newest opening at hour 135
    status, body = post(f"{server}/api/live", {"symbol": "BTCUSDT", "context": 128, "horizon": 6, "since": 3_600_000 * 133})
    assert status == 200
    assert body["new_bars"]["time"] == [3_600_000 * 134, 3_600_000 * 135]
    assert len(body["live"]["bands"]) == 6 and len(body["future_time"]) == 6
    assert body["future_time"][0] == body["last_time"] + 3_600_000


def test_markets_and_yahoo_symbols(server):
    _, _, raw = get(f"{server}/api/markets")
    markets = json.loads(raw)
    assert set(markets) == {"binance", "yahoo"} and "1d" in markets["yahoo"]["intervals"]
    _, _, raw = get(f"{server}/api/symbols?source=yahoo")
    syms = [s["symbol"] for s in json.loads(raw)["symbols"]]
    assert "^NDX" in syms and "XU100.IS" in syms


def test_parse_request_by_source():
    r = parse_request({"source": "yahoo", "symbol": "xu100.is", "interval": "1d"})
    assert r["symbol"] == "XU100.IS" and r["source"] == "yahoo"
    with pytest.raises(ValueError):
        parse_request({"source": "yahoo", "symbol": "AAPL", "interval": "4h"})
    with pytest.raises(ValueError):
        parse_request({"source": "nasdaq", "symbol": "AAPL"})


class SlowForecaster(FakeForecaster):
    def forecast(self, context, prediction_length, past_covariates=None, known_covariates=None):
        time.sleep(0.3)
        return super().forecast(context, prediction_length, past_covariates, known_covariates)


@pytest.fixture()
def slow_server():
    app = App(
        ModelCache("cpu", loader=lambda model_id, device: SlowForecaster()),
        klines_fetcher=fake_klines,
        symbols_fetcher=lambda: [],
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_queue_and_cancel(slow_server):
    body = {"symbol": "BTCUSDT", "bars": 2000, "horizon": 8, "max_origins": 2000, "context": 256}
    _, first = post(f"{slow_server}/api/backtest", body)
    _, second = post(f"{slow_server}/api/backtest", body)
    for _ in range(100):
        jobs = json.loads(get(f"{slow_server}/api/jobs")[2])["jobs"]
        phases = {j["job_id"]: j["phase"] for j in jobs}
        if phases.get(first["job_id"]) == "forecasting" and phases.get(second["job_id"]) == "waiting for model":
            break
        time.sleep(0.05)
    waiting = next(j for j in jobs if j["job_id"] == second["job_id"])
    assert waiting["position"] == 1 and "BTCUSDT" in waiting["label"]
    # cancel the waiting one, then the running one
    for job_id in (second["job_id"], first["job_id"]):
        status, _ = post(f"{slow_server}/api/jobs/{job_id}/cancel", {})
        assert status == 200
        assert wait_job(slow_server, job_id)["status"] == "cancelled"
    # the model is free again: a new job runs to completion
    _, third = post(f"{slow_server}/api/backtest", {**body, "max_origins": 20})
    assert wait_job(slow_server, third["job_id"])["status"] == "done"

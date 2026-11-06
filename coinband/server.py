"""Stdlib HTTP server: a small JSON API around the backtest engine, plus the static UI.

    GET  /api/markets          data sources (Binance, Yahoo), their intervals and defaults
    GET  /api/symbols?source=  symbols for a source (Binance spot pairs, Yahoo presets)
    GET  /api/models           Timestral checkpoints and which are loaded
    POST /api/backtest         start a backtest job, returns {"job_id"}
    POST /api/leaderboard      same settings over several symbols, returns {"job_id"}
    GET  /api/jobs             the queue: running, waiting and recent jobs (no results)
    GET  /api/jobs/<id>        job status, progress and (once done) the result
    POST /api/jobs/<id>/cancel stop a job, whether waiting, fetching or forecasting
    POST /api/live             new closed bars since `since`, a fresh live forecast and the current price
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coinband import __version__
from coinband import cache
from coinband.backtest import BacktestParams, run_backtest
from coinband.binance_data import BinanceError
from coinband.markets import Source, default_sources
from coinband.sessions import future_times

STATIC_DIR = Path(__file__).parent / "static"

MODELS = [
    {"id": "mistralai/Timestral-5M", "name": "Timestral-5M", "size_mb": 21},
    {"id": "mistralai/Timestral-28M", "name": "Timestral-28M", "size_mb": 111},
    {"id": "mistralai/Timestral-123M", "name": "Timestral-123M", "size_mb": 490},
    {"id": "mistralai/Timestral-209M", "name": "Timestral-209M", "size_mb": 837},
]
DEFAULT_MODEL = "mistralai/Timestral-209M"

LIMITS = {
    "bars": (100, 50_000),
    "context": (32, 32_768),
    "horizon": (1, 512),
    "stride": (1, 10_000),
    "max_origins": (1, 10_000),
    "trend_window": (0, 5_000),
}
MAX_LEADERBOARD_SYMBOLS = 20
JOB_TTL_S = 3600
MAX_JOBS = 50


def pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    import torch

    from coinband.device import mps_available

    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


class ModelCache:
    """Loads each checkpoint once. `inference_lock` serializes forecasts so jobs do not fight for the CPU/GPU."""

    def __init__(self, device: str, loader=None) -> None:
        self.device = device
        self._models: dict[str, object] = {}
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._loader = loader

    def loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def get(self, model_id: str):
        with self._load_lock:
            if model_id not in self._models:
                if self._loader is not None:
                    self._models[model_id] = self._loader(model_id, self.device)
                else:
                    from coinband.device import load_forecaster

                    self._models[model_id] = load_forecaster(model_id, self.device)
            return self._models[model_id]


class JobCancelled(Exception):
    pass


ACTIVE = ("queued", "running", "cancelling")


class Job:
    def __init__(self, request: dict, kind: str = "backtest") -> None:
        self.kind = kind
        self.cancel_event = threading.Event()
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"  # queued | running | cancelling | done | error | cancelled
        self.phase = "queued"
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.result: dict | None = None
        self.created = time.time()
        self.lock = threading.Lock()

    def update(self, **fields) -> None:
        self.check()
        with self.lock:
            for k, v in fields.items():
                setattr(self, k, v)

    def check(self) -> None:
        """Raise `JobCancelled` once a cancel was requested; called at every progress step."""
        if self.cancel_event.is_set():
            raise JobCancelled()

    def cancel(self) -> None:
        with self.lock:
            if self.status in ACTIVE:
                self.status = "cancelling"
        self.cancel_event.set()

    def label(self) -> str:
        r = self.request
        target = r.get("symbol") or ", ".join(r.get("symbols") or [])
        model = str(r.get("model", "")).split("/")[-1]
        return f"{self.kind} · {target} · {r.get('interval')} · {model}"

    def snapshot(self, with_result: bool = True) -> dict:
        with self.lock:
            out = {
                "job_id": self.id,
                "kind": self.kind,
                "label": self.label(),
                "created": int(self.created * 1000),
                "status": self.status,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "error": self.error,
            }
            if self.status == "done" and with_result:
                out["result"] = self.result
            return out


def parse_request(body: dict, leaderboard: bool = False, sources: dict[str, Source] | None = None) -> dict:
    """Validate a backtest (or leaderboard) request; raises ValueError with a user-facing message."""
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    sources = sources or default_sources()
    source_name = str(body.get("source", "binance"))
    if source_name not in sources:
        raise ValueError(f"source must be one of {', '.join(sources)}")
    src = sources[source_name]
    if leaderboard:
        raw = body.get("symbols") or src.leaderboard
        if isinstance(raw, str):
            raw = raw.replace(",", " ").split()
        symbols = list(dict.fromkeys(src.parse_symbol(s) for s in raw))
        if not 1 <= len(symbols) <= MAX_LEADERBOARD_SYMBOLS:
            raise ValueError(f"give 1 to {MAX_LEADERBOARD_SYMBOLS} symbols")
        symbol = None
    else:
        symbol = src.parse_symbol(body.get("symbol", ""))
    interval = str(body.get("interval", src.default_interval))
    if interval not in src.intervals:
        raise ValueError(f"interval must be one of {', '.join(src.intervals)} for {src.label}")
    model = str(body.get("model", DEFAULT_MODEL))
    if model not in {m["id"] for m in MODELS}:
        raise ValueError("unknown model")
    defaults = {"bars": 6000, "context": 4096, "horizon": 24, "stride": 1, "max_origins": 1000, "trend_window": 0}
    out: dict = {"source": source_name, "symbol": symbol, "interval": interval, "model": model}
    if leaderboard:
        out["symbols"] = symbols
    for key, default in defaults.items():
        raw = body.get(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
        lo, hi = LIMITS[key]
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        out[key] = value
    cov = body.get("covariates", False)
    if not isinstance(cov, bool):
        raise ValueError("covariates must be true or false")
    out["covariates"] = cov
    if out["trend_window"] == 1:
        raise ValueError("trend_window must be 0 (off) or >= 2")
    since = body.get("since")
    out["since"] = int(since) if isinstance(since, (int, float)) else None
    return out


class App:
    def __init__(
        self,
        models: ModelCache,
        klines_fetcher=None,
        symbols_fetcher=None,
        sources: dict[str, Source] | None = None,
        use_cache: bool = False,
    ) -> None:
        self.models = models
        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        self.sources = sources or default_sources()
        if klines_fetcher is not None:  # test hooks for the Binance source
            self.sources["binance"].fetch = klines_fetcher
        if symbols_fetcher is not None:
            self.sources["binance"].symbols = symbols_fetcher
        self.use_cache = use_cache

    def klines(self, source: str, symbol: str, interval: str, bars: int, progress=None):
        src = self.sources[source]
        if not self.use_cache:
            return src.fetch(symbol, interval, bars, progress=progress)
        return cache.get_klines(source, symbol, interval, bars, src.fetch, src.intervals[interval], progress=progress)

    def future_time(self, klines, horizon: int) -> list[int]:
        src_ms = self.sources_interval_ms(klines)
        info = klines.info or {}
        return future_times(klines.open_time, src_ms, horizon, info.get("timezone"), info.get("continuous", True))

    def sources_interval_ms(self, klines) -> int:
        for src in self.sources.values():
            if klines.interval in src.intervals:
                return src.intervals[klines.interval]
        raise ValueError(f"unknown interval {klines.interval}")

    def submit(self, request: dict, kind: str = "backtest") -> Job:
        job = Job(request, kind)
        with self.jobs_lock:
            now = time.time()
            for jid in [j for j, v in self.jobs.items() if now - v.created > JOB_TTL_S]:
                del self.jobs[jid]
            if len(self.jobs) >= MAX_JOBS:
                oldest = min(self.jobs.values(), key=lambda j: j.created)
                del self.jobs[oldest.id]
            self.jobs[job.id] = job
        target = self._run_leaderboard if kind == "leaderboard" else self._run
        threading.Thread(target=target, args=(job,), daemon=True).start()
        return job

    def job(self, job_id: str) -> Job | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def queue(self) -> list[dict]:
        """Active jobs (oldest first) and the 10 most recent finished ones; `position` counts jobs ahead for the model."""
        with self.jobs_lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created)
        snaps = [j.snapshot(with_result=False) for j in jobs]
        active = [s for s in snaps if s["status"] in ACTIVE]
        finished = [s for s in snaps if s["status"] not in ACTIVE][-10:]
        waiting = [s for s in active if s["phase"] == "waiting for model" or s["status"] == "queued"]
        for s in active:
            s["position"] = waiting.index(s) + 1 if s in waiting else 0
        return active + finished[::-1]

    def cancel(self, job_id: str) -> bool:
        job = self.job(job_id)
        if job is None:
            return False
        job.cancel()
        return True

    def _acquire_model(self, job: Job) -> None:
        """Wait for the inference lock, giving up as soon as the job is cancelled."""
        while not self.models.inference_lock.acquire(timeout=0.25):
            job.check()

    def _backtest(self, job: Job, r: dict, symbol: str, label: str = "") -> tuple:
        """Fetch and backtest one pair, reporting progress on `job`; returns `(klines, result, seconds)`."""
        job.update(phase=f"{label}fetching", done=0, total=r["bars"])
        klines = self.klines(r["source"], symbol, r["interval"], r["bars"], progress=lambda d, t: job.update(done=d, total=t))
        job.update(phase=f"{label}loading model", done=0, total=0)
        forecaster = self.models.get(r["model"])
        params = BacktestParams(
            context=r["context"],
            horizon=r["horizon"],
            stride=r["stride"],
            max_origins=r["max_origins"],
            trend_window=r["trend_window"],
            covariates=r["covariates"],
        )
        job.update(phase=f"{label}waiting for model")
        self._acquire_model(job)
        try:
            job.update(phase=f"{label}forecasting")
            started = time.time()
            bt = run_backtest(
                forecaster,
                klines.close,
                params,
                progress=lambda d, t: job.update(done=d, total=t),
                volume=klines.volume,
            )
            elapsed = time.time() - started
        finally:
            self.models.inference_lock.release()
        return klines, bt, elapsed

    def _guard(self, job: Job, fn) -> None:
        try:
            job.update(status="running")
            result = fn()
            job.update(status="done", phase="done", result=result)
        except JobCancelled:
            with job.lock:
                job.status, job.phase = "cancelled", "cancelled"
        except (BinanceError, ValueError) as e:
            job.update(status="error", phase="error", error=str(e))
        except Exception as e:  # surface anything else to the UI, keep the server alive
            traceback.print_exc()
            job.update(status="error", phase="error", error=f"{type(e).__name__}: {e}")

    def _run(self, job: Job) -> None:
        r = job.request

        def go():
            klines, bt, elapsed = self._backtest(job, r, r["symbol"])
            bt.pop("raw", None)
            return {
                "source": r["source"],
                "symbol": klines.symbol,
                "info": klines.info,
                "interval": klines.interval,
                "interval_ms": self.sources[r["source"]].intervals[klines.interval],
                "future_time": self.future_time(klines, r["horizon"]),
                "model": r["model"],
                "params": {k: r[k] for k in ("context", "horizon", "stride", "max_origins", "trend_window", "covariates")},
                "time": klines.open_time,
                "close": klines.close,
                "high": klines.high,
                "low": klines.low,
                "inference_s": round(elapsed, 2),
                **bt,
            }

        self._guard(job, go)

    def _run_leaderboard(self, job: Job) -> None:
        r = job.request

        def go():
            rows = []
            symbols = r["symbols"]
            for n, symbol in enumerate(symbols):
                label = f"{symbol} ({n + 1}/{len(symbols)}): "
                try:
                    klines, bt, elapsed = self._backtest(job, r, symbol, label)
                except (BinanceError, ValueError) as e:
                    rows.append({"symbol": symbol, "error": str(e)})
                    continue
                s = dict(bt["summary"])
                s.pop("by_horizon", None)
                s.pop("calibration", None)
                plain = s.pop("plain", None)
                live = bt["origins"][-1]
                last = klines.close[-1]
                tip = live["bands"][-1]
                rows.append(
                    {
                        "symbol": klines.symbol,
                        "name": (klines.info or {}).get("name"),
                        "bars": len(klines),
                        "last": last,
                        "summary": s,
                        "plain_crps_skill": plain.get("crps_skill") if plain else None,
                        "live_change": tip[2] / last - 1 if tip[2] is not None else None,
                        "live_band": [tip[1], tip[3]],
                        "inference_s": round(elapsed, 2),
                    }
                )
            return {
                "source": r["source"],
                "interval": r["interval"],
                "model": r["model"],
                "params": {
                    k: r[k] for k in ("bars", "context", "horizon", "stride", "max_origins", "trend_window", "covariates")
                },
                "rows": rows,
            }

        self._guard(job, go)

    def live(self, r: dict) -> dict:
        """New closed bars after `r["since"]`, the live forecast from the newest one and the current price."""
        src = self.sources[r["source"]]
        klines = self.klines(r["source"], r["symbol"], r["interval"], r["context"] + 8)
        forecaster = self.models.get(r["model"])
        params = BacktestParams(
            context=r["context"],
            horizon=r["horizon"],
            stride=1,
            max_origins=1,
            min_context=1,
            trend_window=r["trend_window"],
            covariates=r["covariates"],
        )
        if not self.models.inference_lock.acquire(timeout=20):
            raise ValueError("the model is busy with another job; the live forecast will retry")
        try:
            bt = run_backtest(forecaster, klines.close, params, volume=klines.volume)
        finally:
            self.models.inference_lock.release()
        since = r["since"]
        new = [i for i, t in enumerate(klines.open_time) if since is None or t > since]
        if since is None:
            new = new[-1:]
        try:
            price = src.current_price(r["symbol"])
        except (BinanceError, ValueError, KeyError):
            price = {"price": None, "time": None}
        return {
            "symbol": klines.symbol,
            "new_bars": {
                "time": [klines.open_time[i] for i in new],
                "close": [klines.close[i] for i in new],
                "high": [klines.high[i] for i in new],
                "low": [klines.low[i] for i in new],
            },
            "last_time": klines.open_time[-1],
            "live": bt["origins"][-1],
            "future_time": self.future_time(klines, r["horizon"]),
            "price": price.get("price"),
            "price_time": price.get("time"),
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"coinband/{__version__}"

        def log_message(self, fmt, *args):  # quieter than the default: skip job polling
            if "/api/jobs/" not in self.path:
                super().log_message(fmt, *args)

        def _json(self, payload, status: int = 200) -> None:
            data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/symbols":
                from urllib.parse import parse_qs, urlsplit

                source = parse_qs(urlsplit(self.path).query).get("source", ["binance"])[0]
                if source not in app.sources:
                    self._error(HTTPStatus.BAD_REQUEST, "unknown source")
                    return
                try:
                    self._json({"source": source, "symbols": app.sources[source].symbols()})
                except BinanceError as e:
                    self._error(HTTPStatus.BAD_GATEWAY, str(e))
            elif path == "/api/markets":
                self._json(
                    {
                        name: {
                            "label": src.label,
                            "intervals": list(src.intervals),
                            "default_symbol": src.default_symbol,
                            "default_interval": src.default_interval,
                            "leaderboard": src.leaderboard,
                        }
                        for name, src in app.sources.items()
                    }
                )
            elif path == "/api/models":
                self._json(
                    {
                        "default": DEFAULT_MODEL,
                        "device": app.models.device,
                        "models": [{**m, "loaded": app.models.loaded(m["id"])} for m in MODELS],
                    }
                )
            elif path == "/api/jobs":
                self._json({"jobs": app.queue()})
            elif path.startswith("/api/jobs/"):
                job = app.job(path.rsplit("/", 1)[-1])
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "no such job")
                else:
                    self._json(job.snapshot())
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
            elif path.rstrip("/") in ("/paper", "/paper/risk", "/paper/options"):
                from coinband.paper import paper_html

                page = paper_html(path.rstrip("/").split("/")[-1] if path.rstrip("/") != "/paper" else "paper")
                if page is None:
                    self._error(HTTPStatus.NOT_FOUND, "docs/paper.md not found")
                else:
                    self._send(page.encode(), "text/html; charset=utf-8")
            elif path.startswith("/paper/figures/"):
                from coinband.paper import figure

                target = figure(path.rsplit("/", 1)[-1])
                if target is None:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                else:
                    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self._send(target.read_bytes(), ctype)
            else:
                self._static(path)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                if app.cancel(job_id):
                    self._json({"job_id": job_id, "cancelled": True})
                else:
                    self._error(HTTPStatus.NOT_FOUND, "no such job")
                return
            if path not in ("/api/backtest", "/api/leaderboard", "/api/live"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            kind = {"/api/leaderboard": "leaderboard", "/api/live": "live"}.get(path, "backtest")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                request = parse_request(body, leaderboard=kind == "leaderboard", sources=app.sources)
            except (ValueError, json.JSONDecodeError) as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
                return
            if kind == "live":
                try:
                    self._json(app.live(request))
                except (BinanceError, ValueError) as e:
                    self._error(HTTPStatus.BAD_GATEWAY, str(e))
                return
            job = app.submit(request, kind)
            self._json({"job_id": job.id}, HTTPStatus.ACCEPTED)

        def _send(self, data: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _static(self, path: str) -> None:
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            data = target.read_bytes()
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype == "application/javascript":
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="coinband", description="Backtest Timestral on Binance coins in a browser.")
    parser.add_argument("--host", default=os.environ.get("COINBAND_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("COINBAND_PORT", "8000")))
    parser.add_argument(
        "--device", default=os.environ.get("COINBAND_DEVICE", "auto"), help="auto, cpu, cuda or mps (default: auto)"
    )
    args = parser.parse_args(argv)

    app = App(ModelCache(pick_device(args.device)), use_cache=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"coinband {__version__} on http://{args.host}:{args.port}  (device: {app.models.device})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

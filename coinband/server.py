"""Stdlib HTTP server: a small JSON API around the backtest engine, plus the static UI.

    GET  /api/symbols          spot pairs trading on Binance
    GET  /api/models           Timestral checkpoints and which are loaded
    POST /api/backtest         start a backtest job, returns {"job_id"}
    GET  /api/jobs/<id>        job status, progress and (once done) the result
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
from coinband.backtest import BacktestParams, run_backtest
from coinband.binance_data import INTERVALS, BinanceError, fetch_klines, list_symbols

STATIC_DIR = Path(__file__).parent / "static"

MODELS = [
    {"id": "mistralai/Timestral-5M", "name": "Timestral-5M", "size_mb": 21},
    {"id": "mistralai/Timestral-28M", "name": "Timestral-28M", "size_mb": 111},
    {"id": "mistralai/Timestral-123M", "name": "Timestral-123M", "size_mb": 490},
    {"id": "mistralai/Timestral-209M", "name": "Timestral-209M", "size_mb": 837},
]
DEFAULT_MODEL = "mistralai/Timestral-28M"

LIMITS = {
    "bars": (100, 50_000),
    "context": (32, 32_768),
    "horizon": (1, 512),
    "stride": (1, 10_000),
    "max_origins": (1, 2_000),
}
JOB_TTL_S = 3600
MAX_JOBS = 50


def pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
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
                    from timestral import TimestralForecaster

                    self._models[model_id] = TimestralForecaster.from_pretrained(model_id, device=self.device)
            return self._models[model_id]


class Job:
    def __init__(self, request: dict) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"  # queued | running | done | error
        self.phase = "queued"
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.result: dict | None = None
        self.created = time.time()
        self.lock = threading.Lock()

    def update(self, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            out = {
                "job_id": self.id,
                "status": self.status,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "error": self.error,
            }
            if self.status == "done":
                out["result"] = self.result
            return out


def parse_request(body: dict) -> dict:
    """Validate a backtest request; raises ValueError with a user-facing message."""
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    symbol = str(body.get("symbol", "")).strip().upper()
    if not symbol.isalnum() or not 2 <= len(symbol) <= 20:
        raise ValueError("symbol must be a Binance pair like BTCUSDT")
    interval = str(body.get("interval", "1h"))
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {', '.join(INTERVALS)}")
    model = str(body.get("model", DEFAULT_MODEL))
    if model not in {m["id"] for m in MODELS}:
        raise ValueError("unknown model")
    defaults = {"bars": 3000, "context": 2048, "horizon": 24, "stride": 24, "max_origins": 400}
    out: dict = {"symbol": symbol, "interval": interval, "model": model}
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
    return out


class App:
    def __init__(self, models: ModelCache, klines_fetcher=fetch_klines, symbols_fetcher=list_symbols) -> None:
        self.models = models
        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        self.fetch_klines = klines_fetcher
        self.list_symbols = symbols_fetcher

    def submit(self, request: dict) -> Job:
        job = Job(request)
        with self.jobs_lock:
            now = time.time()
            for jid in [j for j, v in self.jobs.items() if now - v.created > JOB_TTL_S]:
                del self.jobs[jid]
            if len(self.jobs) >= MAX_JOBS:
                oldest = min(self.jobs.values(), key=lambda j: j.created)
                del self.jobs[oldest.id]
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def job(self, job_id: str) -> Job | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def _run(self, job: Job) -> None:
        r = job.request
        try:
            job.update(status="running", phase="fetching", done=0, total=r["bars"])
            klines = self.fetch_klines(
                r["symbol"], r["interval"], r["bars"], progress=lambda d, t: job.update(done=d, total=t)
            )
            job.update(phase="loading model", done=0, total=0)
            forecaster = self.models.get(r["model"])
            params = BacktestParams(
                context=r["context"], horizon=r["horizon"], stride=r["stride"], max_origins=r["max_origins"]
            )
            job.update(phase="waiting for model")
            with self.models.inference_lock:
                job.update(phase="forecasting")
                started = time.time()
                bt = run_backtest(
                    forecaster, klines.close, params, progress=lambda d, t: job.update(done=d, total=t)
                )
                elapsed = time.time() - started
            result = {
                "symbol": klines.symbol,
                "interval": klines.interval,
                "interval_ms": INTERVALS[klines.interval],
                "model": r["model"],
                "params": {k: r[k] for k in ("context", "horizon", "stride", "max_origins")},
                "time": klines.open_time,
                "close": klines.close,
                "high": klines.high,
                "low": klines.low,
                "inference_s": round(elapsed, 2),
                **bt,
            }
            job.update(status="done", phase="done", result=result)
        except (BinanceError, ValueError) as e:
            job.update(status="error", phase="error", error=str(e))
        except Exception as e:  # surface anything else to the UI, keep the server alive
            traceback.print_exc()
            job.update(status="error", phase="error", error=f"{type(e).__name__}: {e}")


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
                try:
                    self._json({"symbols": app.list_symbols()})
                except BinanceError as e:
                    self._error(HTTPStatus.BAD_GATEWAY, str(e))
            elif path == "/api/models":
                self._json(
                    {
                        "default": DEFAULT_MODEL,
                        "device": app.models.device,
                        "models": [{**m, "loaded": app.models.loaded(m["id"])} for m in MODELS],
                    }
                )
            elif path.startswith("/api/jobs/"):
                job = app.job(path.rsplit("/", 1)[-1])
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "no such job")
                else:
                    self._json(job.snapshot())
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
            else:
                self._static(path)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/backtest":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                request = parse_request(body)
            except (ValueError, json.JSONDecodeError) as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
                return
            job = app.submit(request)
            self._json({"job_id": job.id}, HTTPStatus.ACCEPTED)

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

    app = App(ModelCache(pick_device(args.device)))
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

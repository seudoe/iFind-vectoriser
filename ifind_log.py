"""iFind shared server log format (same file in vectoriser, internScraper and resume-extractor-v3 `rx3/ifind_log.py`; the Node side is
ifind/lib/logger.ts).

    14:02:11.532 │ INFO  │ vectoriser │ app │ models ready  tfidf_vocab=12044 ms=1830
    ^dim           ^bold    ^bold magenta ^italic blue  ^plain    ^italic key = coloured value

Levels: DEBUG (dim) INFO (cyan) OK (green) WARN (yellow) ERROR (bold white on red). Colour is off when stderr is not a TTY or NO_COLOR is
set; FORCE_COLOR=1 forces it. Credentials (DB URIs, API keys, bearer tokens) are redacted.

    from ifind_log import setup, get_logger, RequestLogMiddleware
    setup("vectoriser")                      # once, at import time of the entrypoint
    log = get_logger("app")
    log.info("models ready", extra={"f": {"ms": 1830}})        # or log.ok(...), log.warn(...)
    app.add_middleware(RequestLogMiddleware)                   # one line per HTTP request
"""

import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

OK = 25
logging.addLevelName(OK, "OK")


def _ok(self, msg, *args, **kw):
    if self.isEnabledFor(OK):
        self._log(OK, msg, args, **kw)


logging.Logger.ok = _ok  # type: ignore[attr-defined]

_SERVICE = "app"
_STREAM = sys.stderr


def _color_on() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return os.environ["FORCE_COLOR"] != "0"
    return hasattr(_STREAM, "isatty") and _STREAM.isatty()


def _p(code: str, s: str, on: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if on else s


_LEVELS = {
    logging.DEBUG: ("DEBUG", "2;37"),
    logging.INFO: ("INFO ", "1;36"),
    OK: ("OK   ", "1;32"),
    logging.WARNING: ("WARN ", "1;33"),
    logging.ERROR: ("ERROR", "1;97;41"),
    logging.CRITICAL: ("FATAL", "1;97;41"),
}

_REDACT = [
    (re.compile(r"(mongodb(?:\+srv)?://)([^:@/\s]+):([^@\s]+)@", re.I), r"\1\2:***@"),
    (re.compile(r"\b(gsk_|sk-|AIza|ghp_|hf_)[A-Za-z0-9_-]{8,}"), r"\1***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{8,}", re.I), r"\1***"),
]


def redact(text: str) -> str:
    for rx, rep in _REDACT:
        text = rx.sub(rep, text)
    return text


def _value(key: str, v: object, on: bool) -> str:
    s = str(v) if not isinstance(v, str) or not re.search(r"\s", v) else repr(v)
    if key == "status":
        n = int(v) if str(v).isdigit() else 0
        return _p("1;31" if n >= 500 else "33" if n >= 400 else "36" if n >= 300 else "32", s, on)
    if key == "ms":
        return _p("33" if float(v) > 1000 else "32", f"{s}ms", on)  # type: ignore[arg-type]
    if key == "method":
        return _p("1;35", s, on)
    return _p("97", s, on)


class Formatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        on = _color_on()
        label, code = _LEVELS.get(record.levelno, (record.levelname[:5].ljust(5), "1"))
        sep = _p("2", "│", on)
        t = datetime.fromtimestamp(record.created)
        stamp = t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"
        scope = "uvicorn" if record.name.startswith("uvicorn") else record.name
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            msg = _p("1", msg, on)
        fields = getattr(record, "f", None) or {}
        kv = " ".join(f"{_p('2;3', k + '=', on)}{_value(k, v, on)}" for k, v in fields.items() if v is not None)
        parts = [_p("2", stamp, on), _p(code, f" {label} ", on), _p("1;35", self.service, on), _p("3;34", scope, on), msg + (f"  {kv}" if kv else "")]
        line = f" {sep} ".join(parts)
        if record.exc_info:
            line += "\n" + _p("2;31", self.formatException(record.exc_info), on)
        return redact(line)


def setup(service: str, level: str | None = None) -> None:
    """Install the iFind format on the root logger and on uvicorn/fastapi; uvicorn's own access log is replaced by RequestLogMiddleware."""
    global _SERVICE
    _SERVICE = service
    h = logging.StreamHandler(_STREAM)
    h.setFormatter(Formatter(service))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel((level or os.environ.get("LOG_LEVEL", "INFO")).upper())
    for name in ("uvicorn", "uvicorn.error", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").handlers[:] = []
    logging.getLogger("uvicorn.access").propagate = False
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class RequestLogMiddleware:
    """Pure-ASGI access log: `METHOD path status ms bytes client`. Health checks log at DEBUG so probes don't flood the console."""

    QUIET = {"/health", "/"}

    def __init__(self, app):
        self.app = app
        self.log = logging.getLogger("http")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        info = {"status": 500, "bytes": 0}

        async def _send(message):
            if message["type"] == "http.response.start":
                info["status"] = message["status"]
            elif message["type"] == "http.response.body":
                info["bytes"] += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            ms = round((time.perf_counter() - started) * 1000)
            st = info["status"]
            lvl = logging.ERROR if st >= 500 else logging.WARNING if st >= 400 else (logging.DEBUG if scope["path"] in self.QUIET else logging.INFO)
            client = (scope.get("client") or ("-",))[0]
            self.log.log(lvl, "request", extra={"f": {"method": scope["method"], "path": scope["path"], "status": st, "ms": ms, "bytes": info["bytes"], "client": client}})


def capture_print(default_scope: str = "print") -> None:
    """Routes bare `print()` calls (scrapers, scripts, third-party code) through the formatter, so nothing bypasses the log format.
    Only prints to the default stdout are captured; print(file=...) is untouched. A leading `[tag]` becomes the scope; a leading
    ERROR/WARN keyword sets the level."""
    import builtins

    if getattr(builtins.print, "_ifind", False):
        return
    real_print = builtins.print

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is not None and file is not sys.stdout:
            return real_print(*args, sep=sep, end=end, file=file, flush=flush)
        text = sep.join(str(a) for a in args).strip()
        if not text or set(text) <= set("=-─═ >*#_"):
            return  # blank lines and banner rules add nothing in a structured log
        m = re.match(r"^\[([^\]]{1,40})\]\s*(.*)$", text, re.S)
        scope, text = (m.group(1), m.group(2)) if m else (default_scope, text)
        up = text[:12].upper().lstrip("[ ")
        level = logging.ERROR if up.startswith(("ERROR", "FATAL", "TRACEBACK")) else logging.WARNING if up.startswith(("WARN", "SKIP")) else logging.INFO
        for line in text.splitlines():
            logging.getLogger(scope).log(level, line)

    _print._ifind = True  # type: ignore[attr-defined]
    builtins.print = _print

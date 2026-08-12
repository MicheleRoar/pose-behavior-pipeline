"""
local_media_server.py
=======================
Serves arbitrary local video files over http://127.0.0.1:<port>/... with
HTTP Range support, so a `<video>` element inside a pywebview window can
play them.

Why not `<video src="file://...">` (what compare.js did until this):
works fine on macOS's WKWebView backend, but pywebview's Linux backends
(WebKitGTK/QtWebEngine) enforce a stricter policy and refuse to load a
file:// resource referenced from a page itself served over file://,
raising "Not allowed to load local resource" (Michele, 2026-08, real
Linux/CUDA machine -- the webui itself already opens as a file:// URL,
see webui_app.py). Routing through a tiny local HTTP server sidesteps
this platform difference entirely: an http:// URL has no such
restriction on any backend.

Why NOT a simpler fix (read the whole file, hand it to JS as a base64
`data:` URL): would also dodge the file:// restriction, but browsers
can't issue a Range request into a `data:` URL -- the ENTIRE file would
have to be loaded into memory (both in Python and again in the webview)
before playback could start, and scrubbing would be pointless (no
partial fetch possible). A tiny Range-aware HTTP server is barely more
code and gives real streaming/seeking.

Security note: only files EXPLICITLY registered via `register()` (i.e.
ones the user picked through the native file dialog, see
webui/api.py::CompareApi.pick_video_path) are ever servable -- the
filesystem path is never taken from the request/URL itself, so this
can't be used to browse or read arbitrary files. Binds to 127.0.0.1
only (never 0.0.0.0): this must stay unreachable from outside the
machine, it exists to work around a webview quirk, not to be a real
media server.
"""

from __future__ import annotations

import http.server
import re
import threading
from pathlib import Path

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",  # the default output of common/video_writer.py, see there
}
_CHUNK_SIZE = 1024 * 1024  # 1 MiB per read/write -- streams the file instead
# of holding a whole (possibly large) response in memory at once.


class _Registry:
    """token -> real filesystem path. Shared between the HTTP handler
    (a new instance per request, so it can't hold state itself) and
    whoever calls `register()` -- see `LocalMediaServer`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: dict[str, Path] = {}
        self._next_id = 1

    def register(self, path: str) -> str:
        with self._lock:
            token = str(self._next_id)
            self._next_id += 1
            self._paths[token] = Path(path)
        return token

    def resolve(self, token: str) -> Path | None:
        with self._lock:
            return self._paths.get(token)


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    registry: _Registry  # bound per-server, see LocalMediaServer.__init__

    def log_message(self, format, *args) -> None:  # noqa: A002 -- matches base class's signature
        pass  # silences the default per-request stderr line, just noise here

    def do_GET(self) -> None:  # noqa: N802 -- name required by BaseHTTPRequestHandler
        token = self.path.strip("/")
        path = self.registry.resolve(token)
        if path is None or not path.is_file():
            self.send_error(404, "unknown or no longer available file")
            return

        file_size = path.stat().st_size
        content_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")

        start, end = 0, file_size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            match = _RANGE_RE.match(range_header)
            if match:
                status = 206
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                # an open-ended range ("bytes=1000-") is common for a
                # <video> element probing metadata/seeking -- keep
                # `end` at file_size - 1 in that case (already the
                # default above).

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # the <video> element aborted this request (seeked
                    # elsewhere mid-transfer, or the window was closed)
                    # -- not a real error, just stop sending.
                    return
                remaining -= len(chunk)


class LocalMediaServer:
    """One instance per app run, created lazily on first use (see
    `CompareApi`). Binds to an OS-assigned free port on 127.0.0.1 --
    never a fixed port (avoids clashing with anything already running
    on the machine) and never 0.0.0.0 (see module docstring)."""

    def __init__(self) -> None:
        self._registry = _Registry()
        handler = type("_BoundRangeRequestHandler", (_RangeRequestHandler,),
                        {"registry": self._registry})
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._port

    def url_for(self, path: str) -> str:
        """Registers `path` (a real filesystem path) and returns the
        http://127.0.0.1:<port>/<token> URL a <video> element can load
        it from."""
        token = self._registry.register(path)
        return f"http://127.0.0.1:{self._port}/{token}"

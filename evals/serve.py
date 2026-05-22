"""Tiny dev server for the eval viewer + runs."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CorsHandler(SimpleHTTPRequestHandler):
    """Adds permissive CORS headers so viewer can fetch from the same port."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the eval viewer locally")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default=str(Path(__file__).parent))
    ns = parser.parse_args()

    import os

    os.chdir(ns.root)
    server = ThreadingHTTPServer(("127.0.0.1", ns.port), CorsHandler)
    print(f"Serving {ns.root} at http://127.0.0.1:{ns.port}/viewer/")
    print(f"Use http://127.0.0.1:{ns.port}/viewer/?run=<run_id> to auto-load a run.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

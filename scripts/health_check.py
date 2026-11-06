#!/usr/bin/env python3
"""Container HEALTHCHECK / manual probe: exit 0 only when GET /ping returns 200 (model loaded AND engine ready).

204 (initialising) and anything else -> exit 1, so use a generous --start-period (model download + warm-up).
Usage: health_check.py [url]
"""
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    port = os.environ.get("PORT_HEALTH") or os.environ.get("PORT") or "8080"
    url = sys.argv[1] if len(sys.argv) > 1 else f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            code = r.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:
        print(f"unreachable: {exc}")
        return 1
    print({200: "ready", 204: "initializing"}.get(code, f"unhealthy ({code})"))
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())

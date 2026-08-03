from __future__ import annotations

import ctypes
import json
import os
import socket
import sys
import urllib.request
import webbrowser
from pathlib import Path


APP_NAME = "Band Lyric Sync"
DEFAULT_PORT = 7860


def _is_band_lyric_sync_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/config",
            timeout=1.0,
        ) as response:
            payload = json.load(response)
        return payload.get("title") == "Band Lyric Sync Tool"
    except Exception:
        return False


def _available_port(preferred: int) -> int:
    if _is_band_lyric_sync_running(preferred):
        return preferred
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


def _show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(
        0,
        message,
        f"{APP_NAME} - Error",
        0x10,
    )


def main() -> None:
    distribution_root = Path(__file__).resolve().parents[1]
    app_root = distribution_root / "app"
    data_root = Path(os.environ["LOCALAPPDATA"]) / "BandLyricSync"
    log_root = data_root / "logs"
    cache_root = data_root / "cache"
    log_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ["BAND_LYRIC_SYNC_DATA_DIR"] = str(data_root)
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["PATH"] = (
        str(distribution_root / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    open_browser = os.environ.get("BAND_LYRIC_SYNC_OPEN_BROWSER", "1") != "0"
    port = _available_port(DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}"
    if _is_band_lyric_sync_running(port):
        if open_browser:
            webbrowser.open(url)
        return

    os.environ["BAND_LYRIC_SYNC_PORT"] = str(port)
    os.environ["BAND_LYRIC_SYNC_OPEN_BROWSER"] = "1" if open_browser else "0"
    os.chdir(app_root)
    sys.path.insert(0, str(app_root))

    log_file = open(
        log_root / "server.log",
        "a",
        encoding="utf-8",
        buffering=1,
    )
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        import app

        app.build_app().launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=open_browser,
            show_error=True,
            allowed_paths=[str(data_root)],
        )
    except Exception as exc:
        print(f"Startup failed: {type(exc).__name__}: {exc}")
        _show_error(
            "프로그램을 시작하지 못했습니다.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"로그: {log_root / 'server.log'}"
        )
        raise


if __name__ == "__main__":
    main()

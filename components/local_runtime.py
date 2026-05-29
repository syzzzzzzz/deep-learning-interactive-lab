"""Local development server and Streamlit runtime helpers."""

from __future__ import annotations

import http.server
import inspect
import socket
import subprocess
import sys
from functools import partial
from pathlib import Path


DEFAULT_STREAMLIT_PORT = 8501
DEFAULT_STATIC_PORTS = (8000, 8001, 8002, 8003, 4173, 5173, 5500)


def port_is_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_static_port(
    requested_port: int | None = None,
    candidate_ports: tuple[int, ...] = DEFAULT_STATIC_PORTS,
) -> int:
    if requested_port is not None:
        if port_is_available(requested_port):
            return requested_port
        raise RuntimeError(f"端口 {requested_port} 不可用，请换一个端口或关闭占用程序。")

    for port in candidate_ports:
        if port_is_available(port):
            return port
    ports = ", ".join(str(port) for port in candidate_ports)
    raise RuntimeError(f"端口 {ports} 都不可用，请先关闭占用这些端口的程序。")


def run_static_site(project_root: Path, port: int | None = None) -> None:
    selected_port = choose_static_port(port)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(project_root))

    class StaticSiteServer(http.server.ThreadingHTTPServer):
        allow_reuse_address = True

    print("启动静态 HTML 学习网站")
    print(f"地址: http://127.0.0.1:{selected_port}")
    print("请保持这个终端窗口打开；关闭窗口后页面会断开。")
    print("主站入口是 index.html；Streamlit 只保留为 legacy 模块工具。")
    print("=" * 60)
    with StaticSiteServer(("127.0.0.1", selected_port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n网站已停止。")


def run_streamlit_app(
    project_root: Path,
    module_path: Path,
    port: int = DEFAULT_STREAMLIT_PORT,
) -> None:
    print(f"启动 Streamlit 页面: {module_path.relative_to(project_root)}")
    print(f"地址: http://127.0.0.1:{port}")
    print("请保持这个终端窗口打开；关闭窗口后页面会断开。")
    print("=" * 60)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(module_path),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        check=False,
    )


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            return True
    except Exception:
        pass

    streamlit_frames = ("streamlit\\testing", "streamlit\\runtime\\scriptrunner")
    return any(
        any(marker in frame.filename.replace("/", "\\") for marker in streamlit_frames)
        for frame in inspect.stack()
    )

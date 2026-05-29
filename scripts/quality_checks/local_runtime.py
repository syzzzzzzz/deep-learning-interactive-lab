from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def check_local_runtime_module(context: QualityCheckContext) -> None:
    """Verify local server/runtime helpers are outside the main shell."""

    from components import local_runtime

    failures: list[str] = []
    main_text = context.read_text(Path("main.py"))
    runtime_text = context.read_text(Path("components/local_runtime.py"))

    required_runtime_fragments = [
        "DEFAULT_STREAMLIT_PORT = 8501",
        "DEFAULT_STATIC_PORTS = (8000, 8001, 8002, 8003, 4173, 5173, 5500)",
        "def port_is_available",
        "def choose_static_port",
        "def run_static_site",
        "def run_streamlit_app",
        "def running_under_streamlit",
        "ThreadingHTTPServer",
        "streamlit",
        "--browser.gatherUsageStats=false",
    ]
    for fragment in required_runtime_fragments:
        if fragment not in runtime_text:
            failures.append(f"components/local_runtime.py 缺少本地运行契约：{fragment}")

    forbidden_main_fragments = [
        "import http.server",
        "import socket",
        "import inspect",
        "ThreadingHTTPServer",
        "def port_is_available",
        "def choose_static_port",
        "def running_under_streamlit",
    ]
    for fragment in forbidden_main_fragments:
        if fragment in main_text:
            failures.append(f"main.py 仍保留本地运行实现细节：{fragment}")

    required_main_fragments = [
        "from components.local_runtime import",
        "run_local_static_site(BASE_DIR, port)",
        "run_local_streamlit_app(BASE_DIR, module_path, port)",
        "run_static_site(args.port)",
    ]
    for fragment in required_main_fragments:
        if fragment not in main_text:
            failures.append(f"main.py 未委派到 local_runtime：{fragment}")

    if local_runtime.DEFAULT_STATIC_PORTS != (8000, 8001, 8002, 8003, 4173, 5173, 5500):
        failures.append("DEFAULT_STATIC_PORTS 默认候选端口被意外改变")
    if local_runtime.DEFAULT_STREAMLIT_PORT != 8501:
        failures.append("DEFAULT_STREAMLIT_PORT 默认端口被意外改变")
    if local_runtime.choose_static_port(0) != 0:
        failures.append("choose_static_port(0) 应接受系统随机端口请求")

    if failures:
        raise QualityCheckFailure("本地运行模块检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 本地运行模块检查：端口选择、静态站启动和 Streamlit 运行判断已由 local_runtime 承担")

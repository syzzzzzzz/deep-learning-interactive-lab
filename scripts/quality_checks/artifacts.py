from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from .common import QualityCheckContext, QualityCheckFailure


ROOT_RUNTIME_ARTIFACT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".csv",
    ".pt",
    ".pth",
    ".ckpt",
    ".log",
}
ROOT_RUNTIME_TEMP_DIRS = {
    "__pycache__",
    "tmp",
    "temp",
    "outputs",
    "output",
    "runs",
}
ROOT_RUNTIME_ALLOWED_DIRS = {
    ".git",
    ".streamlit_module_outputs",
    ".streamlit",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "venv",
    "env",
}


def check_root_runtime_artifacts_clean(context: QualityCheckContext) -> None:
    """Keep direct script products out of the repository root."""

    failures: list[str] = []
    for path in sorted(context.root.iterdir(), key=lambda item: item.name.lower()):
        if path.name in ROOT_RUNTIME_ALLOWED_DIRS:
            continue
        if path.is_dir() and path.name in ROOT_RUNTIME_TEMP_DIRS:
            failures.append(f"{path.name}/")
        elif path.is_file() and path.suffix.lower() in ROOT_RUNTIME_ARTIFACT_EXTENSIONS:
            failures.append(path.name)

    if failures:
        raise QualityCheckFailure(
            "根目录运行产物污染检查失败：发现脚本输出落在项目根目录。\n"
            "请把图片、CSV、模型权重、日志、缓存和临时输出统一放入 artifacts 或 legacy run 目录。\n"
            + "\n".join(f"  - {item}" for item in failures)
        )
    print("[通过] 根目录运行产物污染检查：根目录无图片、CSV、模型权重、日志、缓存或临时输出")


def check_direct_script_artifact_redirection(context: QualityCheckContext) -> None:
    """Ensure direct legacy script runs are redirected into the shared artifact root."""

    text = context.read_text(Path("sitecustomize.py"))
    required_fragments = [
        "ARTIFACT_ROOT",
        "ROOT_ARTIFACT_RUN_DIR",
        "ARTIFACT_DIR_ENV",
        "DL_BOOK_ARTIFACT_DIR",
        "sys.dont_write_bytecode = True",
        "_cleanup_root_pycache",
        "builtins.open = redirected_open",
        "Path.open = redirected_path_open",
        "Path.write_text = redirected_path_write_text",
        "Path.write_bytes = redirected_path_write_bytes",
        "plt.savefig = redirected_pyplot_savefig",
        "Figure.savefig = redirected_figure_savefig",
        "torch.save = redirected_torch_save",
        "pd.DataFrame.to_csv = redirected_to_csv",
        '".streamlit_module_outputs"',
        '".csv"',
        '".log"',
    ]
    failures = [fragment for fragment in required_fragments if fragment not in text]

    runner_text = context.read_text(Path("legacy_runner.py"))
    main_text = context.read_text(Path("main.py"))
    legacy_page_text = context.read_text(Path("components/legacy_page.py"))
    runtime_text = context.read_text(Path("components/artifact_runtime.py"))
    runtime_fragments = [
        "class ArtifactRun",
        "def output_root",
        "def safe_output_root",
        "def latest_run_dir",
        "def build_subprocess_env",
        "def create_run_dir",
        "def write_run_status",
        "def write_run_streams",
        "def run_legacy_script",
        "def read_run_text",
        "def run_status",
        "def image_artifacts",
        "def artifact_context",
        "DL_BOOK_ARTIFACT_DIR",
        "PYTHONDONTWRITEBYTECODE",
    ]
    for fragment in runtime_fragments:
        if fragment not in runtime_text:
            failures.append(f"components/artifact_runtime.py 缺少运行产物契约：{fragment}")
    if "DL_BOOK_ARTIFACT_DIR" not in runner_text:
        failures.append("legacy_runner.py 缺少 DL_BOOK_ARTIFACT_DIR run 目录传递")
    if "import sitecustomize" not in runner_text:
        failures.append("legacy_runner.py 缺少显式导入 sitecustomize")
    if "from components.legacy_page import LegacyPageDeps" not in main_text:
        failures.append("main.py 未委派旧脚本页面到 components.legacy_page")
    if "artifact_context(run_dir)" not in legacy_page_text:
        failures.append("components/legacy_page.py 仍未通过 artifact_context 消费产物上下文")
    if "run_legacy_script(deps.project_root" not in legacy_page_text:
        failures.append("components/legacy_page.py 仍未通过 run_legacy_script 执行旧脚本")

    if failures:
        raise QualityCheckFailure(
            "直接运行脚本产物重定向检查失败：sitecustomize.py 缺少关键保护。\n"
            + "\n".join(f"  - {fragment}" for fragment in failures)
        )
    print("[通过] 直接运行脚本产物重定向检查：图片、CSV、模型权重、日志和 pycache 均有统一保护")


def check_artifact_runtime_behavior(context: QualityCheckContext) -> None:
    """Exercise the artifact runtime through its public interface."""

    from components.artifact_runtime import (
        ARTIFACT_DIR_ENV,
        artifact_context,
        build_subprocess_env,
        create_run_dir,
        latest_run_dir,
        output_root,
        run_status,
        safe_output_root,
        write_run_status,
        write_run_streams,
    )

    module_target = "quality_checks/artifact_runtime_behavior"
    run_dir = create_run_dir(context.root, module_target)
    failures: list[str] = []

    try:
        expected_root = output_root(context.root)
        expected_module_root = safe_output_root(context.root, module_target)
        if not str(run_dir).startswith(str(expected_module_root)):
            failures.append("create_run_dir 未把 run 放入对应模块的安全产物目录")
        if latest_run_dir(context.root, module_target) != run_dir:
            failures.append("latest_run_dir 未返回刚创建的最新 run")

        env = build_subprocess_env(run_dir)
        if env.get("PYTHONIOENCODING") != "utf-8":
            failures.append("build_subprocess_env 未固定 UTF-8 输出")
        if env.get("PYTHONDONTWRITEBYTECODE") != "1":
            failures.append("build_subprocess_env 未禁止 __pycache__ 写入")
        if env.get("MPLBACKEND") != "Agg":
            failures.append("build_subprocess_env 未固定 Matplotlib 后端")
        if env.get(ARTIFACT_DIR_ENV) != str(run_dir):
            failures.append("build_subprocess_env 未把 DL_BOOK_ARTIFACT_DIR 指向当前 run")

        write_run_streams(run_dir, "stdout-ok", "stderr-ok")
        write_run_status(run_dir, 0, False)
        return_code, timed_out = run_status(run_dir)
        if return_code != 0 or timed_out:
            failures.append("run_status 未正确读取 return_code/timed_out")

        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
            "53de0000000c49444154789c63606060000000040001f61738550000000049"
            "454e44ae426082"
        )
        (run_dir / "a.png").write_bytes(png_bytes)
        (run_dir / "b.png").write_bytes(png_bytes)
        context_data = artifact_context(run_dir)
        if context_data["stdout"] != "stdout-ok":
            failures.append("artifact_context 未读取 stdout")
        if context_data["stderr"] != "stderr-ok":
            failures.append("artifact_context 未读取 stderr")
        if context_data["return_code"] != 0:
            failures.append("artifact_context 未读取 return_code")
        if context_data["image_count"] != 1:
            failures.append("artifact_context 未对重复图片去重")
        if not expected_root.exists():
            failures.append("output_root 未创建或定位统一产物根目录")
    finally:
        for file_path in run_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        run_dir.rmdir()
        module_root = safe_output_root(context.root, module_target)
        if module_root.exists() and not any(module_root.iterdir()):
            module_root.rmdir()

    if failures:
        raise QualityCheckFailure("运行产物 runtime 行为检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 运行产物 runtime 行为检查：run 生命周期、环境变量、状态流和图片去重均正常")


def check_legacy_script_artifact_run(context: QualityCheckContext) -> None:
    """Run a tiny legacy script and verify all products stay in the run dir."""

    from components.artifact_runtime import artifact_context, run_legacy_script, safe_output_root

    module_target = "quality_checks/legacy_script_artifact_run"
    module_root = safe_output_root(context.root, module_target)
    probe_dir = context.root / ".streamlit_module_outputs" / "_quality_probe"
    probe_script = probe_dir / "legacy_artifact_probe.py"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_script.write_text(
        dedent(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt

            print("artifact probe stdout")
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.title("artifact probe")
            plt.show()
            Path("probe.log").write_text("log redirected", encoding="utf-8")
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = run_legacy_script(context.root, module_target, probe_script, timeout_seconds=20)
    data = artifact_context(result.run_dir)
    failures: list[str] = []
    try:
        if result.return_code != 0:
            failures.append(f"探针脚本返回码异常：{result.return_code}")
        if "artifact probe stdout" not in str(data["stdout"]):
            failures.append("stdout 未写入 run 目录")
        if int(data["image_count"]) < 1:
            failures.append("Matplotlib 图片未写入 run 目录")
        if not (result.run_dir / "probe.log").exists():
            failures.append("普通日志文件未重定向到 run 目录")
        if (context.root / "probe.log").exists():
            failures.append("普通日志文件污染了项目根目录")
        if not str(result.run_dir).startswith(str(module_root)):
            failures.append("run_legacy_script 未使用模块安全产物目录")
    finally:
        for file_path in result.run_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        result.run_dir.rmdir()
        if module_root.exists() and not any(module_root.iterdir()):
            module_root.rmdir()
        probe_script.unlink(missing_ok=True)
        if probe_dir.exists() and not any(probe_dir.iterdir()):
            probe_dir.rmdir()

    if failures:
        raise QualityCheckFailure("旧脚本产物 run 集成检查失败：\n" + "\n".join(f"  - {item}" for item in failures))
    print("[通过] 旧脚本产物 run 集成检查：stdout、图片、日志和 run 目录生命周期均正常")

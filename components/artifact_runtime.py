"""Runtime artifact lifecycle for legacy teaching scripts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_ROOT_NAME = ".streamlit_module_outputs"
ARTIFACT_DIR_ENV = "DL_BOOK_ARTIFACT_DIR"


@dataclass(frozen=True)
class ArtifactRun:
    run_dir: Path
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


def output_root(project_root: Path) -> Path:
    return project_root / ARTIFACT_ROOT_NAME


def safe_output_root(project_root: Path, module_target: str) -> Path:
    digest = hashlib.sha1(module_target.encode("utf-8")).hexdigest()[:12]
    return output_root(project_root) / digest


def latest_run_dir(project_root: Path, module_target: str) -> Path | None:
    root = safe_output_root(project_root, module_target)
    if not root.is_dir():
        return None
    runs = [path for path in root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime, default=None)


def build_subprocess_env(artifact_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    if artifact_dir is not None:
        env[ARTIFACT_DIR_ENV] = str(artifact_dir)
    return env


def create_run_dir(project_root: Path, module_target: str) -> Path:
    root = safe_output_root(project_root, module_target)
    root.mkdir(parents=True, exist_ok=True)
    base_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{base_name}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_status(run_dir: Path, return_code: int | None, timed_out: bool) -> None:
    (run_dir / "status.txt").write_text(
        f"return_code={return_code}\ntimed_out={timed_out}\n",
        encoding="utf-8",
        errors="replace",
    )


def write_run_streams(run_dir: Path, stdout: str, stderr: str) -> None:
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")


def run_legacy_script(
    project_root: Path,
    module_target: str,
    script_path: Path,
    timeout_seconds: int = 45,
) -> ArtifactRun:
    run_dir = create_run_dir(project_root, module_target)
    command = [
        sys.executable,
        str(project_root / "legacy_runner.py"),
        str(project_root),
        str(script_path),
        str(run_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=build_subprocess_env(run_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n运行超过 {timeout_seconds} 秒，已停止。"

    write_run_streams(run_dir, stdout, stderr)
    write_run_status(run_dir, return_code, timed_out)
    return ArtifactRun(
        run_dir=run_dir,
        return_code=return_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )


def read_run_text(run_dir: Path, name: str) -> str:
    path = run_dir / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_status(run_dir: Path) -> tuple[int | None, bool]:
    text = read_run_text(run_dir, "status.txt")
    return_code: int | None = None
    timed_out = False
    for line in text.splitlines():
        if line.startswith("return_code="):
            try:
                return_code = int(line.split("=", 1)[1])
            except ValueError:
                return_code = None
        elif line.startswith("timed_out="):
            timed_out = line.split("=", 1)[1].strip().lower() == "true"
    return return_code, timed_out


def image_artifacts(run_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(run_dir.glob(pattern))
    unique: list[Path] = []
    seen_hashes: set[str] = set()
    for image_path in sorted(images, key=lambda path: (path.name.startswith("figure_"), path.name)):
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique.append(image_path)
    return unique


def artifact_context(run_dir: Path) -> dict[str, object]:
    return_code, timed_out = run_status(run_dir)
    images = image_artifacts(run_dir)
    return {
        "run_dir": run_dir,
        "return_code": return_code,
        "timed_out": timed_out,
        "stdout": read_run_text(run_dir, "stdout.txt"),
        "stderr": read_run_text(run_dir, "stderr.txt"),
        "images": images,
        "image_count": len(images),
    }

from __future__ import annotations

from pathlib import Path

from .common import QualityCheckContext, QualityCheckFailure


def require_fragments(text: str, fragments: list[str], owner: str, failures: list[str]) -> None:
    for fragment in fragments:
        if fragment not in text:
            failures.append(f"{owner}: 缺少静态学习体验片段 {fragment}")


def check_navigation_and_learning_ux(context: QualityCheckContext) -> None:
    """Check the static shell's student-facing navigation contract.

    This is intentionally a deeper module than the old all-in-one quality script:
    it owns the user-path invariants for the static site, while the top-level
    runner only decides when to invoke it.
    """

    failures: list[str] = []
    html_text = context.read_text(Path("index.html"))
    js_text = context.read_text(Path("assets/site.js"))
    css_text = context.read_text(Path("assets/site.css"))
    package_text = context.read_text(Path("package.json")) if context.exists(Path("package.json")) else ""
    playwright_text = context.read_text(Path("playwright.config.js")) if context.exists(Path("playwright.config.js")) else ""
    pages_text = context.read_text(Path(".github/workflows/pages.yml")) if context.exists(Path(".github/workflows/pages.yml")) else ""
    readme_text = context.read_text(Path("README.md")) if context.exists(Path("README.md")) else ""
    architecture_text = context.read_text(Path("docs/architecture.md")) if context.exists(Path("docs/architecture.md")) else ""
    teaching_text = context.read_text(Path("docs/teaching_design.md")) if context.exists(Path("docs/teaching_design.md")) else ""
    e2e_text = (
        context.read_text(Path("tests/e2e/course-learning-flow.spec.js"))
        if context.exists(Path("tests/e2e/course-learning-flow.spec.js"))
        else ""
    )

    require_fragments(
        html_text,
        [
            'class="brand" href="#home"',
            'href="#starter"',
            'href="#path"',
            'href="#courses"',
            "打开全站目录",
            "全站课程目录抽屉",
            "已知关键词搜索",
        ],
        "index.html",
        failures,
    )

    require_fragments(
        js_text,
        [
            "hash.startsWith(\"#course/\")",
            "hash.startsWith(\"#console/\")",
            "decodeURIComponent(location.hash",
            "返回首页",
            "上一节",
            "下一节",
            "返回本路径",
            "data-mark-understood",
            "data-mark-review",
            "data-learning-mode",
            "data-tag-filter",
            "drawer-onboarding",
            "course-topline",
            "course-console-cta",
            "mode-switcher",
            "console-purpose-strip",
            "readLearningMode",
            "writeLearningMode",
            "wireTagFilters",
            "wireCourseModeSwitcher",
            "去控制台完成 1 个验证",
        ],
        "assets/site.js",
        failures,
    )

    require_fragments(
        css_text,
        [
            ".side-drawer.is-open",
            ".code-window",
            "resize: vertical",
            ".tag-button",
            ".course-topline",
            ".course-console-cta",
            ".mode-switcher",
            ".drawer-onboarding",
            ".console-purpose-strip",
            ".model-builder-grid",
            ".model-node-list",
            ".shape-diagnostics",
            ".diagnostic-error",
            ".code-export",
            'body[data-learning-mode="beginner"] .course-layout .advanced-only',
        ],
        "assets/site.css",
        failures,
    )

    require_fragments(
        js_text,
        [
            'data-testid="course-layout"',
            'data-testid="lesson-roadmap"',
            'data-testid="lesson-three-minute"',
            'data-testid="course-toc"',
            'data-testid="toc-${target.replace("course-", "")}"',
            'data-testid="developer-source"',
            'data-testid="mode-beginner"',
            'data-testid="mode-advanced"',
            'data-testid="learning-actions"',
            'data-testid="mark-understood"',
            'data-testid="mark-review"',
            'data-testid="learning-status"',
            'data-testid="next-lesson"',
            "renderImmediateThreeMinuteBrief",
            'data-testid="console-builder"',
            'data-testid="model-preset"',
            'data-testid="load-preset"',
            'data-testid="model-node-list"',
            'data-testid="shape-diagnostics"',
            'data-testid="export-code"',
            'data-testid="code-export"',
            'data-testid="save-graph"',
            'data-testid="graph-json"',
            'data-testid="load-graph"',
            "MODEL_LAYER_LIBRARY",
            "MODEL_PRESETS",
            "transformer_mini",
            "shape_error",
            "generatePyTorchCode",
            "inferModelGraph",
            "wireModelBuilder",
        ],
        "assets/site.js",
        failures,
    )

    require_fragments(
        package_text,
        [
            '"screenshots": "node scripts/capture_screenshots.mjs"',
            '"test:e2e": "playwright test"',
            '"@playwright/test"',
        ],
        "package.json",
        failures,
    )
    require_fragments(
        playwright_text,
        [
            "webServer",
            "python -m http.server 8877 --bind 127.0.0.1",
            "chromium-desktop",
            "mobile-chrome",
        ],
        "playwright.config.js",
        failures,
    )
    require_fragments(
        e2e_text,
        [
            "student can enter a lesson, follow the guide, mark progress, and continue",
            "beginner and advanced modes keep source code out of the beginner path",
            "getByTestId(\"lesson-roadmap\")",
            "getByTestId(\"toc-animation\")",
            "localStorage.getItem(\"deep-learning-book-progress-v1\")",
        ],
        "tests/e2e/course-learning-flow.spec.js",
        failures,
    )
    require_fragments(
        e2e_text + (context.read_text(Path("tests/e2e/central-console-builder.spec.js")) if context.exists(Path("tests/e2e/central-console-builder.spec.js")) else ""),
        [
            "loads transformer preset, diagnoses shapes, exports code, and restores saved graph",
            "getByTestId(\"console-builder\")",
            "selectOption(\"transformer_mini\")",
            "nn.MultiheadAttention",
            "selectOption(\"shape_error\")",
        ],
        "tests/e2e/central-console-builder.spec.js",
        failures,
    )
    require_fragments(
        pages_text,
        [
            "Deploy GitHub Pages",
            "actions/configure-pages@v5",
            "actions/upload-pages-artifact@v4",
            "actions/deploy-pages@v4",
            "path: _site",
            "cp -R part7_interview _site/part7_interview",
        ],
        ".github/workflows/pages.yml",
        failures,
    )
    require_fragments(
        readme_text,
        [
            "https://syzzzzzzz.github.io/deep-learning-interactive-lab/",
            "docs/screenshots/home.png",
            "docs/screenshots/console-builder.png",
            "docs/architecture.md",
            "docs/teaching_design.md",
            "Node 24",
        ],
        "README.md",
        failures,
    )
    require_fragments(
        architecture_text,
        [
            "components/course_manifest.py",
            "components/legacy_runtime.py",
            "components/artifact_runtime.py",
            "中央控制台",
            "Playwright E2E",
            ".github/workflows/pages.yml",
        ],
        "docs/architecture.md",
        failures,
    )
    require_fragments(
        teaching_text,
        [
            "3 分钟版",
            "新手模式",
            "进阶模式",
            "shape 诊断",
            "八股文训练营",
        ],
        "docs/teaching_design.md",
        failures,
    )
    for screenshot in [
        "docs/screenshots/home.png",
        "docs/screenshots/course-transformer.png",
        "docs/screenshots/console-builder.png",
        "docs/screenshots/shape-diagnostic.png",
        "docs/screenshots/interview-camp.png",
    ]:
        if not context.exists(Path(screenshot)):
            failures.append(f"{screenshot}: 缺少作品集截图")

    forbidden_fragments = [
        "Course File",
        "Personal Learning System",
        "Central Practice Console",
        "Zero Basics",
    ]
    for fragment in forbidden_fragments:
        if fragment in html_text or fragment in js_text:
            failures.append(f"静态站仍有旧学习路径/开发者文案残留：{fragment}")

    if failures:
        raise QualityCheckFailure("静态站导航与学习体验检查失败：\n" + "\n".join(failures))
    print("[通过] 静态站导航与学习体验检查：任务化入口、标签筛选、模式切换和课程上下文均正常")

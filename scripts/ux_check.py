from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import websocket


ROOT = Path(__file__).resolve().parents[1]
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

PAGES = [
    {
        "name": "home-desktop",
        "path": "/",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "home",
    },
    {
        "name": "home-mobile",
        "path": "/",
        "width": 390,
        "height": 844,
        "mobile": True,
        "kind": "home",
    },
    {
        "name": "math-primer",
        "path": "/#course/part1%2Fmath_primer",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
    {
        "name": "transformer",
        "path": "/#course/part4%2Ftransformer_models",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
    {
        "name": "central-console",
        "path": "/#console/part4%2Ftransformer_models",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "console",
    },
    {
        "name": "course-mobile",
        "path": "/#course/part2%2F02_feature_maps",
        "width": 390,
        "height": 844,
        "mobile": True,
        "kind": "course",
    },
    {
        "name": "sequence",
        "path": "/#course/part3%2Fsequence_models",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
    {
        "name": "training",
        "path": "/#course/part5%2Fdata_training",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
    {
        "name": "architecture",
        "path": "/#course/part6%2Ffrontier",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
    {
        "name": "systems",
        "path": "/#course/part7%2Fnetworking",
        "width": 1366,
        "height": 900,
        "mobile": False,
        "kind": "course",
    },
]


def edge_path() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise RuntimeError("Microsoft Edge was not found in the usual install paths.")


def wait_for_cdp(port: int) -> None:
    deadline = time.time() + 12
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1).raise_for_status()
            return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Edge CDP did not start on port {port}.")


class Cdp:
    def __init__(self, port: int):
        self.ws: websocket.WebSocket | None = None
        self.seq = 0
        self.events: list[dict[str, Any]] = []
        last_error: Exception | None = None

        for _ in range(5):
            try:
                tab = self.open_tab(port)
                self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10)
                self.call("Page.enable")
                self.call("Runtime.enable")
                self.call("Log.enable")
                return
            except Exception as exc:
                last_error = exc
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
                time.sleep(0.45)

        raise RuntimeError(f"Unable to connect to Edge CDP tab: {last_error}")

    @staticmethod
    def open_tab(port: int) -> dict[str, Any]:
        new_url = f"http://127.0.0.1:{port}/json/new?about%3Ablank"
        response = requests.put(new_url, timeout=3)
        if response.ok:
            return response.json()
        tabs = requests.get(f"http://127.0.0.1:{port}/json/list", timeout=3).json()
        return next(tab for tab in tabs if tab.get("type") == "page")

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("CDP websocket is not connected")
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.seq:
                return message
            self.events.append(message)

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        payload = result.get("result", {}).get("result", {})
        if "value" in payload:
            return payload["value"]
        return payload

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()


HOME_EVAL = r"""
(async () => {
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const colorOf = (selector, prop) => {
    const el = document.querySelector(selector);
    return el ? getComputedStyle(el)[prop] : "";
  };
  const rectOf = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)};
  };

  await sleep(260);
  document.querySelector("[data-open-menu]")?.click();
  await sleep(180);
  const drawerInput = document.querySelector("#drawer-search");
  if (drawerInput) {
    drawerInput.value = "Transformer";
    drawerInput.dispatchEvent(new Event("input", {bubbles: true}));
  }
  await sleep(120);
  const drawerOpened = document.querySelector("[data-drawer]")?.classList.contains("is-open") || false;
  const drawerLinks = [...document.querySelectorAll(".drawer-link")].map(el => el.innerText.trim());
  document.querySelector("[data-close-menu]")?.click();
  const appShell = document.querySelector("#app");
  const card = document.querySelector(".course-card");
  const drawer = document.querySelector("[data-drawer]");
  const portfolio = document.querySelector("#portfolio");
  const hardcore = document.querySelector("#hardcore-labs");
  const llmCookbook = document.querySelector("#llm-cookbook");
  const hardcoreLabs = [...document.querySelectorAll("[data-hardcore-lab]")];
  const hardcoreBefore = hardcoreLabs.map(lab => lab.querySelector("[data-hardcore-readout]")?.innerText || "");
  hardcoreLabs.forEach((lab) => {
    const control = lab.querySelector("[data-hardcore-control]");
    if (!control) return;
    if (control.tagName === "SELECT") {
      control.selectedIndex = (control.selectedIndex + 1) % control.options.length;
    } else if (control.type === "range") {
      const min = Number(control.min || 0);
      const max = Number(control.max || 100);
      const current = Number(control.value || min);
      control.value = String(current < (min + max) / 2 ? max : min);
    }
    control.dispatchEvent(new Event("input", {bubbles: true}));
    control.dispatchEvent(new Event("change", {bubbles: true}));
  });
  await sleep(160);
  const hardcoreAfter = hardcoreLabs.map(lab => lab.querySelector("[data-hardcore-readout]")?.innerText || "");

  return {
    title: document.title,
    hash: location.hash,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    textLength: (document.body.innerText || "").length,
    bodyBg: getComputedStyle(document.body).backgroundColor,
    bodyColor: getComputedStyle(document.body).color,
    h1: document.querySelector("h1")?.innerText || "",
    heroRect: rectOf(".hero"),
    heroMediaRect: rectOf(".hero-media"),
    portfolioRect: rectOf("#portfolio"),
    hardcoreRect: rectOf("#hardcore-labs"),
    portfolioCardCount: document.querySelectorAll("#portfolio .profile-card").length,
    techStackCount: document.querySelectorAll("#portfolio .tech-stack span").length,
    codeWallCount: document.querySelectorAll("#portfolio .code-wall article").length,
    hardcoreCardCount: document.querySelectorAll("#hardcore-labs .hardcore-lab-card").length,
    hardcoreControlCount: document.querySelectorAll("#hardcore-labs [data-hardcore-control]").length,
    hardcoreStageCount: document.querySelectorAll("#hardcore-labs [data-hardcore-stage]").length,
    hardcoreMetricCount: document.querySelectorAll("#hardcore-labs .hardcore-meter").length,
    hardcoreReadoutCount: document.querySelectorAll("#hardcore-labs [data-hardcore-readout]").length,
    hardcoreLinkCount: document.querySelectorAll("#hardcore-labs .hardcore-links a").length,
    xaiCellCount: document.querySelectorAll("#hardcore-labs .xai-cell").length,
    sampleCellCount: document.querySelectorAll("#hardcore-labs .sample-cell").length,
    challengeChecklistCount: document.querySelectorAll("#hardcore-labs .challenge-checklist span").length,
    caseNodeCount: document.querySelectorAll("#hardcore-labs .case-node").length,
    hardcoreReadoutChangedCount: hardcoreAfter.filter((text, index) => text && text !== hardcoreBefore[index]).length,
    portfolioTextLength: portfolio ? portfolio.innerText.length : 0,
    hardcoreTextLength: hardcore ? hardcore.innerText.length : 0,
    llmCookbookCardCount: document.querySelectorAll("#llm-cookbook .llm-track-card").length,
    llmCookbookDetailCount: document.querySelectorAll("#llm-cookbook .llm-detail-list li").length,
    llmCookbookTextLength: llmCookbook ? llmCookbook.innerText.length : 0,
    llmCookbookSourceHref: document.querySelector("#llm-cookbook .source-note a")?.getAttribute("href") || "",
    courseCards: document.querySelectorAll(".course-card").length,
    catalogCards: document.querySelectorAll("#catalog-grid .module-card").length,
    hasDrawer: !!document.querySelector("[data-drawer]"),
    drawerOpened,
    drawerSearchMatches: drawerLinks.length,
    drawerSearchSample: drawerLinks.slice(0, 5),
    motionRevealCount: document.querySelectorAll(".motion-reveal").length,
    visibleRevealCount: document.querySelectorAll(".motion-reveal.is-visible").length,
    appAnimation: appShell ? getComputedStyle(appShell).animationName : "",
    cardTransition: card ? getComputedStyle(card).transitionProperty : "",
    drawerTransition: drawer ? getComputedStyle(drawer).transitionProperty : "",
    navDisplay: colorOf(".nav-links", "display"),
    menuButtonRect: rectOf(".menu-button"),
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    visibleWidth: window.innerWidth,
  };
})()
"""


COURSE_EVAL = r"""
(async () => {
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const rectOf = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: Math.round(r.x), y: Math.round(r.y), right: Math.round(r.right), w: Math.round(r.width), h: Math.round(r.height)};
  };
  const code = document.querySelector(".code-window");
  const codeStyle = code ? getComputedStyle(code) : null;
  const source = document.querySelector("[data-source]");
  const homeLinks = [...document.querySelectorAll("a")]
    .filter(el => (el.getAttribute("href") || "") === "#home")
    .map(el => (el.innerText || el.getAttribute("aria-label") || "").trim());
  const mutateControl = (control) => {
    if (!control) return;
    if (control.tagName === "SELECT") {
      control.selectedIndex = (control.selectedIndex + 1) % control.options.length;
    } else if (control.type === "checkbox") {
      control.checked = !control.checked;
    } else if (control.type === "range") {
      const min = Number(control.min || 0);
      const max = Number(control.max || 100);
      const current = Number(control.value || min);
      control.value = String(current < (min + max) / 2 ? max : min);
    }
    control.dispatchEvent(new Event("input", {bubbles: true}));
    control.dispatchEvent(new Event("change", {bubbles: true}));
  };
  const demo = document.querySelector("[data-demo]");
  const demoReadout = demo?.querySelector("[data-demo-readout]");
  const demoControls = demo ? [...demo.querySelectorAll("[data-demo-control]")] : [];
  const demoReadoutBefore = demoReadout?.innerText || "";
  const lab = document.querySelector("[data-lab]");
  const readout = lab?.querySelector("[data-lab-readout]");
  const controls = lab ? [...lab.querySelectorAll("input, select")] : [];
  const readoutBefore = readout?.innerText || "";

  if (demoControls.length) {
    mutateControl(demoControls[0]);
    await sleep(180);
  }

  if (controls.length) {
    mutateControl(controls[0]);
    await sleep(180);
  }

  const trace = document.querySelector(".motion-trace");
  const point = document.querySelector(".motion-point");
  const matrixCell = document.querySelector(".matrix-cell");
  const attentionBar = document.querySelector(".attention-row i");
  const conceptAttentionBar = demo?.querySelector(".attention-score-row i");
  const sequenceMemoryBar = lab?.querySelector("[data-memory-bars] i");
  const demoFlowLine = document.querySelector(".demo-flow-line");
  const scanWindow = document.querySelector(".scan-grid i");
  const memoryBar = document.querySelector(".memory-lane i");
  const demoNode = document.querySelector(".flow-node, .arch-node, .system-node");
  const lessonNotes = document.querySelector("[data-lesson-notes]");
  const demoStage = demo?.querySelector("[data-demo-stage]");
  const attentionMechanism = demo?.querySelector("[data-attention-mechanism]");
  const dryGoods = document.querySelector("[data-dry-goods]");
  const knowledgePointIndex = document.querySelector("[data-knowledge-points]");
  const llmCookbook = document.querySelector("[data-llm-cookbook]");
  const zeroBasics = document.querySelector("[data-zero-basics]");
  const zeroBasicsHeadings = zeroBasics
    ? [...zeroBasics.querySelectorAll("h2, h3")].map(el => el.innerText.trim())
    : [];
  const zeroBasicsAction = zeroBasics?.querySelector(".zero-basics-action .action");
  const lessonCode = document.querySelector(".lesson-code");
  const lessonCodeStyle = lessonCode ? getComputedStyle(lessonCode) : null;
  const appShell = document.querySelector("#app");

  return {
    title: document.title,
    hash: location.hash,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    textLength: (document.body.innerText || "").length,
    h1: document.querySelector("h1")?.innerText || "",
    hasCourseLayout: !!document.querySelector(".course-layout"),
    hasAside: !!document.querySelector(".course-aside"),
    homeLinks,
    lessonCardCount: document.querySelectorAll(".lesson-card").length,
    dryGoodsCardCount: document.querySelectorAll(".dry-goods-card").length,
    dryGoodsTextLength: dryGoods ? (dryGoods.innerText || "").length : 0,
    zeroBasicsCount: document.querySelectorAll(".zero-basics-card").length,
    zeroBasicsTextLength: zeroBasics ? (zeroBasics.innerText || "").length : 0,
    zeroBasicsHeadings,
    zeroBasicsActionHref: zeroBasicsAction?.getAttribute("href") || "",
    conceptDemoCount: document.querySelectorAll("[data-demo]").length,
    demoKind: demo?.dataset.demo || "",
    demoControlCount: demoControls.length,
    demoReadoutBefore,
    demoReadoutAfter: demoReadout?.innerText || "",
    demoStageTextLength: demoStage ? (demoStage.innerText || "").length : 0,
    demoControlsRect: rectOf(".concept-demo .lab-controls"),
    demoStageRect: rectOf("[data-demo-stage]"),
    demoStageClientWidth: demoStage ? demoStage.clientWidth : 0,
    demoStageScrollWidth: demoStage ? demoStage.scrollWidth : 0,
    attentionMechanismClientWidth: attentionMechanism ? attentionMechanism.clientWidth : 0,
    attentionMechanismScrollWidth: attentionMechanism ? attentionMechanism.scrollWidth : 0,
    flowNodeCount: document.querySelectorAll(".flow-node").length,
    conceptSvgCount: document.querySelectorAll(".concept-svg").length,
    attentionMechanismCount: demo ? demo.querySelectorAll("[data-attention-mechanism]").length : 0,
    attentionScoreRowCount: demo ? demo.querySelectorAll(".attention-score-row").length : 0,
    attentionQueryText: demo?.querySelector("[data-attention-query]")?.innerText || "",
    attentionOutputText: demo?.querySelector("[data-attention-output]")?.innerText || "",
    scanCellCount: document.querySelectorAll(".scan-grid span").length,
    memoryBarCount: document.querySelectorAll(".memory-lane i").length,
    demoFlowAnimation: demoFlowLine ? getComputedStyle(demoFlowLine).animationName : "",
    scanAnimation: scanWindow ? getComputedStyle(scanWindow).animationName : "",
    memoryAnimation: memoryBar ? getComputedStyle(memoryBar).animationName : "",
    demoNodeAnimation: demoNode ? getComputedStyle(demoNode).animationName : "",
    lessonNotesLoaded: lessonNotes?.dataset.loaded || "",
    lessonNotesFallback: lessonNotes?.dataset.fallback || "",
    lessonNotesLegacyPath: lessonNotes?.dataset.legacyPath || lessonNotes?.dataset.legacyPath || lessonNotes?.getAttribute("data-legacy-path") || "",
    lessonNotesTextLength: lessonNotes ? (lessonNotes.innerText || "").length : 0,
    knowledgePointIndexTextLength: knowledgePointIndex ? (knowledgePointIndex.innerText || "").length : 0,
    knowledgePointCount: document.querySelectorAll("[data-knowledge-points] li").length,
    knowledgePointHasScroll: knowledgePointIndex ? (knowledgePointIndex.querySelector("ol")?.scrollHeight > knowledgePointIndex.querySelector("ol")?.clientHeight) : false,
    llmCookbookCardCount: llmCookbook ? llmCookbook.querySelectorAll(".llm-track-card").length : 0,
    llmCookbookDetailCount: llmCookbook ? llmCookbook.querySelectorAll(".llm-detail-list li").length : 0,
    llmCookbookTextLength: llmCookbook ? (llmCookbook.innerText || "").length : 0,
    llmCookbookSourceHref: llmCookbook?.querySelector(".source-note a")?.getAttribute("href") || "",
    deepDiveCardCount: document.querySelectorAll(".deep-dive-card").length,
    deepDiveDrillCount: document.querySelectorAll(".deep-dive-drill li").length,
    deepDiveDrillTextLength: [...document.querySelectorAll(".deep-dive-drill")].map(el => el.innerText || "").join("\\n").length,
    lessonOutlineCount: document.querySelectorAll(".lesson-outline span").length,
    lessonCodeCount: document.querySelectorAll(".lesson-code").length,
    lessonCodeText: lessonCode ? (lessonCode.innerText || "") : "",
    lessonCodeResize: lessonCodeStyle ? lessonCodeStyle.resize : "",
    lessonCodeOverflowX: lessonCodeStyle ? lessonCodeStyle.overflowX : "",
    lessonCodeOverflowY: lessonCodeStyle ? lessonCodeStyle.overflowY : "",
    lessonCodeClientHeight: lessonCode ? lessonCode.clientHeight : 0,
    lessonCodeScrollHeight: lessonCode ? lessonCode.scrollHeight : 0,
    knowledgeItemCount: document.querySelectorAll(".knowledge-list li").length,
    practiceTextLength: document.querySelector(".practice-callout")?.innerText.length || 0,
    labType: lab?.dataset.lab || "",
    labRect: rectOf(".interactive-lab"),
    labControlCount: controls.length,
    labReadoutBefore: readoutBefore,
    labReadoutAfter: readout?.innerText || "",
    labReadoutAnimating: readout ? getComputedStyle(readout).animationName : "",
    revealCount: document.querySelectorAll(".motion-reveal").length,
    visibleRevealCount: document.querySelectorAll(".motion-reveal.is-visible").length,
    appAnimation: appShell ? getComputedStyle(appShell).animationName : "",
    traceAnimation: trace ? getComputedStyle(trace).animationName : "",
    pointAnimation: point ? getComputedStyle(point).animationName : "",
    motionPointCount: document.querySelectorAll(".motion-point").length,
    matrixAnimation: matrixCell ? getComputedStyle(matrixCell).animationName : "",
    attentionAnimation: attentionBar ? getComputedStyle(attentionBar).animationName : "",
    conceptAttentionAnimation: conceptAttentionBar ? getComputedStyle(conceptAttentionBar, "::after").animationName : "",
    sequenceMemoryAnimation: sequenceMemoryBar ? getComputedStyle(sequenceMemoryBar).animationName : "",
    matrixCellCount: document.querySelectorAll(".matrix-cell").length,
    attentionRowCount: document.querySelectorAll(".attention-row").length,
    gradientPointCount: document.querySelectorAll("[data-gradient-plot] circle").length,
    sequenceTokenCount: document.querySelectorAll("[data-sequence-tokens] span").length,
    sequenceMemoryCount: document.querySelectorAll("[data-memory-bars] i").length,
    stateCardCount: document.querySelectorAll(".state-card").length,
    metricCardCount: document.querySelectorAll(".metric-card").length,
    trainingPathCount: document.querySelectorAll("[data-training-plot] path").length,
    archNodeCount: document.querySelectorAll("[data-architecture-flow] .arch-node").length,
    systemNodeCount: document.querySelectorAll("[data-systems-flow] .system-node").length,
    codeTextLength: source ? source.innerText.length : 0,
    codeTextSample: source ? source.innerText.slice(0, 80) : "",
    codeRect: rectOf(".code-window"),
    codeScrollWidth: code ? code.scrollWidth : 0,
    codeScrollHeight: code ? code.scrollHeight : 0,
    codeOverflowX: codeStyle ? codeStyle.overflowX : "",
    codeOverflowY: codeStyle ? codeStyle.overflowY : "",
    codeResize: codeStyle ? codeStyle.resize : "",
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    visibleWidth: window.innerWidth,
  };
})()
"""


CONSOLE_EVAL = r"""
(async () => {
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const rectOf = (selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: Math.round(r.x), y: Math.round(r.y), right: Math.round(r.right), w: Math.round(r.width), h: Math.round(r.height)};
  };
  const mutateControl = (control) => {
    if (!control) return;
    if (control.tagName === "SELECT") {
      control.selectedIndex = (control.selectedIndex + 1) % control.options.length;
    } else if (control.type === "checkbox") {
      control.checked = !control.checked;
    } else if (control.type === "range") {
      const min = Number(control.min || 0);
      const max = Number(control.max || 100);
      const current = Number(control.value || min);
      control.value = String(current < (min + max) / 2 ? max : min);
    }
    control.dispatchEvent(new Event("input", {bubbles: true}));
    control.dispatchEvent(new Event("change", {bubbles: true}));
  };

  const lab = document.querySelector("[data-lab]");
  const controls = lab ? [...lab.querySelectorAll("input, select")] : [];
  const readout = lab?.querySelector("[data-lab-readout]");
  const result = document.querySelector("[data-console-result]");
  const note = document.querySelector("[data-console-note]");
  const readoutBefore = readout?.innerText || "";
  const resultBefore = result?.innerText || "";
  const firstNode = document.querySelector("[data-node]");
  const nodeReadout = document.querySelector("[data-node-readout]");
  const nodeBefore = firstNode ? rectOf("[data-node]") : null;

  if (controls.length) {
    mutateControl(controls[0]);
    await sleep(220);
  }

  if (firstNode) {
    const r = firstNode.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const opts = {bubbles: true, cancelable: true, pointerId: 7, pointerType: "mouse", buttons: 1};
    firstNode.dispatchEvent(new PointerEvent("pointerdown", {...opts, clientX: cx, clientY: cy}));
    firstNode.dispatchEvent(new PointerEvent("pointermove", {...opts, clientX: cx + 86, clientY: cy + 42}));
    firstNode.dispatchEvent(new PointerEvent("pointerup", {...opts, buttons: 0, clientX: cx + 86, clientY: cy + 42}));
    await sleep(120);
  }

  const readoutAfter = readout?.innerText || "";
  const resultAfter = result?.innerText || "";
  const nodeAfter = firstNode ? rectOf("[data-node]") : null;
  const noteBefore = note?.value || "";
  document.querySelector("[data-console-fill]")?.click();
  await sleep(80);
  const noteAfter = note?.value || "";
  const appShell = document.querySelector("#app");
  const returnCourseHref = document.querySelector('.console-topline a[href^="#course/"]')?.getAttribute("href") || "";

  return {
    title: document.title,
    hash: location.hash,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    textLength: (document.body.innerText || "").length,
    h1: document.querySelector("h1")?.innerText || "",
    hasCentralConsole: !!document.querySelector("[data-central-console]"),
    moduleId: document.querySelector("[data-central-console]")?.dataset.moduleId || "",
    panelCount: document.querySelectorAll(".console-panel").length,
    chipCount: document.querySelectorAll(".console-chip-list span").length,
    stepCount: document.querySelectorAll(".console-steps li").length,
    hasWorkbench: !!document.querySelector(".console-workbench"),
    hasNodeCanvas: !!document.querySelector("[data-node-canvas]"),
    nodeCount: document.querySelectorAll("[data-node]").length,
    edgeCount: document.querySelectorAll("[data-canvas-edges] line").length,
    nodeBefore,
    nodeAfter,
    nodeReadoutText: nodeReadout?.innerText || "",
    hasTrainingBus: !!document.querySelector("[data-training-bus]"),
    eventLogCount: document.querySelectorAll("[data-event-log] .event-log-item").length,
    eventSubscriberCount: document.querySelectorAll("[data-event-subscriber]").length,
    busLoss: document.querySelector("[data-bus-loss]")?.innerText || "",
    busGradient: document.querySelector("[data-bus-gradient]")?.innerText || "",
    busFeature: document.querySelector("[data-bus-feature]")?.innerText || "",
    busNote: document.querySelector("[data-bus-note]")?.innerText || "",
    labType: lab?.dataset.lab || "",
    labRect: rectOf(".interactive-lab"),
    labControlCount: controls.length,
    labReadoutBefore: readoutBefore,
    labReadoutAfter: readoutAfter,
    consoleMetricCount: document.querySelectorAll("[data-console-metrics] .metric-card").length,
    consoleResultBefore: resultBefore,
    consoleResultAfter: resultAfter,
    noteBefore,
    noteAfter,
    returnCourseHref,
    returnHomeHref: document.querySelector('.console-topline a[href="#home"]')?.getAttribute("href") || "",
    revealCount: document.querySelectorAll(".motion-reveal").length,
    visibleRevealCount: document.querySelectorAll(".motion-reveal.is-visible").length,
    appAnimation: appShell ? getComputedStyle(appShell).animationName : "",
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    visibleWidth: window.innerWidth,
  };
})()
"""


def wait_until(cdp: Cdp, expression: str, description: str, timeout: float = 12) -> None:
    deadline = time.time() + timeout
    last_value: Any = None
    while time.time() < deadline:
        last_value = cdp.evaluate(expression)
        if last_value:
            return
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {description}. Last value: {last_value!r}")


def browser_errors(cdp: Cdp) -> list[str]:
    errors: list[str] = []
    for event in cdp.events:
        method = event.get("method")
        params = event.get("params", {})
        if method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            text = details.get("text") or details.get("exception", {}).get("description")
            errors.append(str(text or "runtime exception"))
        if method == "Log.entryAdded":
            entry = params.get("entry", {})
            if entry.get("level") in {"error", "warning"}:
                errors.append(str(entry.get("text") or entry))
    return errors


def inspect_page(
    base_url: str,
    page: dict[str, Any],
    port: int,
    screenshots: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + page["path"]
    cdp = Cdp(port)
    try:
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": page["width"],
                "height": page["height"],
                "deviceScaleFactor": 1,
                "mobile": page["mobile"],
            },
        )
        cdp.events.clear()
        cdp.call("Page.navigate", {"url": url})
        wait_until(cdp, "document.readyState === 'complete'", "document ready")

        if page["kind"] == "home":
            wait_until(
                cdp,
                "document.querySelector('#catalog-grid')?.children.length > 0 && document.body.innerText.length > 800",
                "home content",
            )
            data = cdp.evaluate(HOME_EVAL)
            failures = validate_home(data, page["mobile"])
        elif page["kind"] == "console":
            wait_until(
                cdp,
                """
                (() => {
                  const consoleRoot = document.querySelector('[data-central-console]');
                  const lab = document.querySelector('[data-lab]');
                  const result = document.querySelector('[data-console-result]');
                  return !!consoleRoot && !!lab && !!result && result.innerText.length > 20;
                })()
                """,
                "central console and native lab",
            )
            data = cdp.evaluate(CONSOLE_EVAL)
            failures = validate_console(data)
        else:
            wait_until(
                cdp,
                """
                (() => {
                  const source = document.querySelector('[data-source]');
                  const lab = document.querySelector('[data-lab]');
                  return !!source && !!lab && !source.innerText.includes('正在读取源码') && source.innerText.length > 200;
                })()
                """,
                "course source preview and native lab",
            )
            data = cdp.evaluate(COURSE_EVAL)
            failures = validate_course(data)

        failures.extend(f"browser error: {error}" for error in browser_errors(cdp))

        if screenshots:
            shot = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            out_dir = Path(tempfile.gettempdir()) / "deep_learning_book_ux_screens"
            out_dir.mkdir(exist_ok=True)
            (out_dir / f"{page['name']}.png").write_bytes(base64.b64decode(shot["result"]["data"]))

        data["name"] = page["name"]
        data["url"] = url
        data["failures"] = failures
        return data
    finally:
        cdp.close()


def validate_home(data: dict[str, Any], mobile: bool) -> list[str]:
    failures: list[str] = []
    if "深度学习书库" not in data.get("title", ""):
        failures.append("document title is not the static learning site")
    if "让学习变得" not in data.get("h1", ""):
        failures.append("home hero headline did not render")
    if data.get("textLength", 0) < 1600:
        failures.append("home page rendered too little text")
    if not data.get("heroMediaRect") or data["heroMediaRect"]["h"] < (260 if mobile else 420):
        failures.append("hero image/media area is missing or collapsed")
    if data.get("courseCards", 0) < 3:
        failures.append("featured course cards are missing")
    if data.get("portfolioCardCount", 0) < 1:
        failures.append("personal technical profile section is missing")
    if data.get("techStackCount", 0) < 4:
        failures.append("technical profile stack is too thin")
    if data.get("codeWallCount", 0) < 4:
        failures.append("code wall does not show enough project artifacts")
    if data.get("hardcoreCardCount", 0) < 4:
        failures.append("hardcore lab zone is missing required lab cards")
    if data.get("hardcoreControlCount", 0) < 12:
        failures.append("hardcore lab zone does not expose enough real controls")
    if data.get("hardcoreStageCount", 0) < 4:
        failures.append("hardcore lab zone is missing experiment stages")
    if data.get("hardcoreMetricCount", 0) < 12:
        failures.append("hardcore lab zone metrics are too thin")
    if data.get("hardcoreReadoutCount", 0) < 4:
        failures.append("hardcore lab zone readouts are missing")
    if data.get("xaiCellCount", 0) < 36:
        failures.append("XAI lab did not render the heatmap grid")
    if data.get("sampleCellCount", 0) < 72:
        failures.append("adversarial lab did not render clean/perturbed samples")
    if data.get("challengeChecklistCount", 0) < 4:
        failures.append("challenge lab did not render a scoring checklist")
    if data.get("caseNodeCount", 0) < 4:
        failures.append("end-to-end case lab did not render the pipeline nodes")
    if data.get("hardcoreReadoutChangedCount", 0) < 2:
        failures.append("hardcore lab controls did not update enough readouts")
    if data.get("hardcoreLinkCount", 0) < 8:
        failures.append("hardcore lab zone does not link to enough concrete lessons")
    if data.get("portfolioTextLength", 0) < 180:
        failures.append("personal technical profile content is too thin")
    if data.get("hardcoreTextLength", 0) < 260:
        failures.append("hardcore lab zone content is too thin")
    if data.get("llmCookbookCardCount", 0) < 8:
        failures.append("LLM cookbook route does not show enough engineering tracks")
    if data.get("llmCookbookDetailCount", 0) < 24:
        failures.append("LLM cookbook route is missing workflow detail steps")
    if data.get("llmCookbookTextLength", 0) < 5000:
        failures.append("LLM cookbook route content is too thin")
    if "datawhalechina/llm-cookbook" not in data.get("llmCookbookSourceHref", ""):
        failures.append("LLM cookbook route is missing the Datawhale source link")
    if data.get("catalogCards", 0) < 20:
        failures.append("catalog did not render enough modules")
    if not data.get("hasDrawer"):
        failures.append("course drawer is missing")
    if not data.get("drawerOpened"):
        failures.append("drawer did not open from the menu button")
    if data.get("drawerSearchMatches", 0) < 3:
        failures.append("drawer search returned too few Transformer matches")
    if data.get("motionRevealCount", 0) < 6:
        failures.append("home page has too few motion reveal targets")
    if data.get("visibleRevealCount", 0) < 2:
        failures.append("home page reveal targets did not become visible")
    if not data.get("reducedMotion"):
        if data.get("appAnimation") in {"", "none"}:
            failures.append("home route entry animation is missing")
        if "transform" not in data.get("cardTransition", ""):
            failures.append("course cards are missing smooth hover motion")
        if "transform" not in data.get("drawerTransition", ""):
            failures.append("drawer slide transition is missing")
    if data.get("bodyBg") not in {"rgb(255, 255, 255)", "rgba(0, 0, 0, 0)"}:
        failures.append(f"body background is not the expected light theme: {data.get('bodyBg')}")
    if data.get("scrollWidth", 0) > data.get("clientWidth", 0) + 4:
        failures.append("home page has horizontal overflow")
    return failures


def validate_course(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not data.get("hash", "").startswith("#course/"):
        failures.append("course route did not keep the expected hash")
    if not data.get("hasCourseLayout"):
        failures.append("course layout did not render")
    if "返回首页" not in data.get("homeLinks", []):
        failures.append("course page is missing a visible return-home link")
    if data.get("lessonCardCount", 0) < 3:
        failures.append("course page is missing the lesson-brief substance cards")
    if data.get("dryGoodsCardCount", 0) < 6:
        failures.append("hard-notes section is missing substance cards")
    if data.get("dryGoodsTextLength", 0) < 1400:
        failures.append("hard-notes section is too thin")
    if data.get("zeroBasicsCount", 0) < 12:
        failures.append("zero-basics section is missing one of the twelve explanation cards")
    required_zero_headings = [
        "零基础解释模板",
        "这是什么？",
        "生活类比",
        "一句话直觉",
        "严谨定义",
        "图中每个元素代表什么",
        "颜色/亮度/方向/速度代表什么",
        "用户应该调哪个参数",
        "观察什么变化",
        "为什么会这样",
        "常见误区",
        "工程用途",
        "去中央控制台实战",
    ]
    zero_headings = set(data.get("zeroBasicsHeadings", []))
    for heading in required_zero_headings:
        if heading not in zero_headings:
            failures.append(f"zero-basics section is missing heading: {heading}")
    if data.get("zeroBasicsTextLength", 0) < 900:
        failures.append("zero-basics section is too thin")
    if not str(data.get("zeroBasicsActionHref", "")).startswith("#console/"):
        failures.append("zero-basics central console action does not link to the practice console")
    if data.get("conceptDemoCount", 0) < 1:
        failures.append("course page is missing the concept animation block")
    if data.get("demoControlCount", 0) < 3:
        failures.append("concept animation has too few controls")
    if len(data.get("demoReadoutAfter", "")) < 20:
        failures.append("concept animation did not render a meaningful readout")
    if data.get("demoReadoutBefore") == data.get("demoReadoutAfter"):
        failures.append("concept animation did not respond to control changes")
    demo_rect = data.get("demoStageRect") or {}
    if demo_rect.get("w", 0) <= 0 or demo_rect.get("h", 0) < 180:
        failures.append("concept animation stage is missing or collapsed")
    controls_rect = data.get("demoControlsRect") or {}
    if controls_rect and demo_rect and controls_rect.get("w", 0) > 0 and demo_rect.get("w", 0) > 0:
        if controls_rect.get("y", 0) + 2 < demo_rect.get("y", 0) + demo_rect.get("h", 0) and demo_rect.get("y", 0) + 2 < controls_rect.get("y", 0) + controls_rect.get("h", 0):
            if demo_rect.get("x", 0) - controls_rect.get("right", 0) < 28:
                failures.append("concept animation controls are too close to the stage")
    if data.get("flowNodeCount", 0) < 3:
        failures.append("concept animation flow nodes are missing")
    if data.get("demoKind") == "attention":
        if data.get("attentionMechanismCount", 0) < 1:
            failures.append("attention concept demo is missing the Q/K/V mechanism chain")
        if data.get("attentionScoreRowCount", 0) < 5:
            failures.append("attention concept demo does not show per-token QK scores")
        attention_text = f"{data.get('attentionQueryText', '')} {data.get('attentionOutputText', '')} {data.get('demoReadoutAfter', '')}"
        for term in ("Query", "Q·K", "softmax", "Value", "权重"):
            if term not in attention_text:
                failures.append(f"attention concept demo is missing evaluable term: {term}")
        if not data.get("reducedMotion") and data.get("conceptAttentionAnimation") in {"", "none"}:
            failures.append("attention concept demo softmax bar motion is missing")
        if data.get("demoStageScrollWidth", 0) > data.get("demoStageClientWidth", 0) + 4:
            failures.append("attention concept demo stage has horizontal overflow")
        if data.get("attentionMechanismScrollWidth", 0) > data.get("attentionMechanismClientWidth", 0) + 4:
            failures.append("attention mechanism content overflows its stage")
    if data.get("lessonNotesLoaded") != "true":
        failures.append("lesson deep-dive notes did not finish loading")
    if data.get("lessonNotesTextLength", 0) < 900:
        failures.append("lesson deep-dive notes are too thin")
    if data.get("lessonNotesFallback") == "generic":
        failures.append("lesson deep-dive fell back to generic notes")
    if data.get("knowledgePointCount", 0) < 10:
        failures.append("knowledge-point full index is missing or too thin")
    if data.get("knowledgePointIndexTextLength", 0) < 700:
        failures.append("knowledge-point full index has too little substance")
    if data.get("knowledgePointHasScroll"):
        failures.append("knowledge-point full index should expand naturally without an inner scrollbar")
    if data.get("hash") in {"#course/part4%2Ftransformer_models", "#course/part5%2Fdata_training", "#course/part7%2Fnetworking"}:
        if data.get("llmCookbookCardCount", 0) < 4:
            failures.append("course page is missing LLM cookbook bridge cards")
        if data.get("llmCookbookDetailCount", 0) < 12:
            failures.append("course LLM cookbook bridge is missing workflow detail steps")
        if data.get("llmCookbookTextLength", 0) < 3000:
            failures.append("course LLM cookbook bridge is too thin")
        if "datawhalechina/llm-cookbook" not in data.get("llmCookbookSourceHref", ""):
            failures.append("course LLM cookbook bridge is missing the source link")
    if data.get("deepDiveCardCount", 0) < 3:
        failures.append("lesson deep-dive cards are missing")
    if data.get("deepDiveDrillCount", 0) < data.get("deepDiveCardCount", 0) * 3:
        failures.append("lesson deep-dive cards are missing actionable study drills")
    if data.get("deepDiveDrillTextLength", 0) < 360:
        failures.append("lesson deep-dive study drills are too thin")
    if data.get("lessonOutlineCount", 0) < 3:
        failures.append("lesson deep-dive outline is missing")
    if data.get("knowledgeItemCount", 0) < 6:
        failures.append("knowledge reading/trap list is too short")
    if data.get("practiceTextLength", 0) < 40:
        failures.append("practice callout is missing or too thin")
    if data.get("lessonCodeCount", 0) and data.get("lessonCodeOverflowX") not in {"auto", "scroll"}:
        failures.append("lesson code snippet cannot scroll horizontally")
    if data.get("lessonCodeCount", 0) and data.get("lessonCodeScrollHeight", 0) > data.get("lessonCodeClientHeight", 0) + 4:
        failures.append("lesson code snippet should expand vertically instead of clipping")
    if data.get("lessonCodeCount", 0) and str(data.get("lessonCodeText", "")).rstrip().endswith("..."):
        failures.append("lesson code snippet should not be truncated with an ellipsis")
    if data.get("lessonNotesFallback") == "source" and str(data.get("lessonCodeText", "")).lstrip().startswith("def css"):
        failures.append("source fallback lesson notes should show a teaching function, not the CSS helper")
    if not data.get("reducedMotion"):
        if data.get("demoNodeAnimation") in {"", "none"}:
            failures.append("concept animation node motion is missing")
        if data.get("demoKind") in {"gradient", "training"} and data.get("demoFlowAnimation") in {"", "none"}:
            failures.append("concept animation trace motion is missing")
        if data.get("demoKind") == "convolution" and data.get("scanAnimation") in {"", "none"}:
            failures.append("concept animation scan motion is missing")
        if data.get("demoKind") == "sequence" and data.get("memoryAnimation") in {"", "none"}:
            failures.append("concept animation memory motion is missing")
    if not data.get("labType"):
        failures.append("course page is missing the native interactive lab")
    if data.get("labControlCount", 0) < 1:
        failures.append("native interactive lab has no controls")
    if len(data.get("labReadoutAfter", "")) < 12:
        failures.append("native interactive lab did not render a readout")
    if data.get("labReadoutBefore") == data.get("labReadoutAfter"):
        failures.append("native interactive lab did not respond to control changes")
    if data.get("revealCount", 0) < 4:
        failures.append("course page has too few motion reveal targets")
    if data.get("visibleRevealCount", 0) < 1:
        failures.append("course page reveal targets did not become visible")
    if not data.get("reducedMotion") and data.get("appAnimation") in {"", "none"}:
        failures.append("course route entry animation is missing")
    if data.get("labType") == "math-gradient" and data.get("gradientPointCount", 0) < 3:
        failures.append("math lab did not render the gradient trajectory")
    if (
        data.get("labType") == "math-gradient"
        and not data.get("reducedMotion")
        and data.get("traceAnimation") in {"", "none"}
    ):
        failures.append("math lab gradient path animation is missing")
    if data.get("labType") == "math-gradient" and data.get("motionPointCount", 0) < 3:
        failures.append("math lab motion points are missing")
    if data.get("labType") == "cnn-feature" and data.get("matrixCellCount", 0) < 25:
        failures.append("CNN lab did not render matrix cells")
    if (
        data.get("labType") == "cnn-feature"
        and not data.get("reducedMotion")
        and data.get("matrixAnimation") in {"", "none"}
    ):
        failures.append("CNN lab matrix cell animation is missing")
    if data.get("labType") == "attention" and data.get("attentionRowCount", 0) < 5:
        failures.append("attention lab did not render token weights")
    if (
        data.get("labType") == "attention"
        and not data.get("reducedMotion")
        and data.get("attentionAnimation") in {"", "none"}
    ):
        failures.append("attention lab bar animation is missing")
    if data.get("labType") == "sequence-memory" and data.get("sequenceTokenCount", 0) < 4:
        failures.append("sequence lab did not render sequence tokens")
    if data.get("labType") == "sequence-memory" and data.get("sequenceMemoryCount", 0) < 4:
        failures.append("sequence lab did not render memory bars")
    if data.get("labType") == "sequence-memory" and data.get("stateCardCount", 0) < 3:
        failures.append("sequence lab did not render state cards")
    if (
        data.get("labType") == "sequence-memory"
        and not data.get("reducedMotion")
        and data.get("sequenceMemoryAnimation") in {"", "none"}
    ):
        failures.append("sequence lab memory animation is missing")
    if data.get("labType") == "training-diagnostics" and data.get("trainingPathCount", 0) < 3:
        failures.append("training lab did not render diagnostic curves")
    if data.get("labType") == "training-diagnostics" and data.get("metricCardCount", 0) < 3:
        failures.append("training lab did not render metric cards")
    if data.get("labType") == "architecture-flow" and data.get("archNodeCount", 0) < 5:
        failures.append("architecture lab did not render architecture nodes")
    if data.get("labType") == "architecture-flow" and data.get("stateCardCount", 0) < 3:
        failures.append("architecture lab did not render architecture metrics")
    if data.get("labType") == "systems-flow" and data.get("systemNodeCount", 0) < 5:
        failures.append("systems lab did not render systems nodes")
    if data.get("labType") == "systems-flow" and data.get("metricCardCount", 0) < 3:
        failures.append("systems lab did not render systems metrics")
    if data.get("codeTextLength", 0) < 1000:
        failures.append("source preview did not load enough code")
    if "暂时无法读取" in data.get("codeTextSample", ""):
        failures.append("source preview failed to fetch the Python source")

    code_rect = data.get("codeRect") or {}
    if code_rect.get("w", 0) <= 0 or code_rect.get("h", 0) <= 0:
        failures.append("code window is missing or collapsed")
    if data.get("codeScrollWidth", 0) > code_rect.get("w", 0) and data.get("codeOverflowX") not in {"auto", "scroll"}:
        failures.append("wide code window cannot scroll horizontally")
    if data.get("codeScrollHeight", 0) > code_rect.get("h", 0) and data.get("codeOverflowY") not in {"auto", "scroll"}:
        failures.append("tall code window cannot scroll vertically")
    if data.get("codeResize") == "none":
        failures.append("code window cannot be resized")
    if data.get("scrollWidth", 0) > data.get("clientWidth", 0) + 4:
        failures.append("course page has horizontal overflow")
    return failures


def validate_console(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not data.get("hash", "").startswith("#console/"):
        failures.append("central console route did not keep the expected hash")
    if "中央控制台实战" not in data.get("h1", ""):
        failures.append("central console headline did not render")
    if not data.get("hasCentralConsole"):
        failures.append("central console root did not render")
    if data.get("moduleId") != "part4/transformer_models":
        failures.append(f"central console did not keep the originating module id: {data.get('moduleId')}")
    if data.get("textLength", 0) < 900:
        failures.append("central console rendered too little instructional text")
    if data.get("panelCount", 0) < 3:
        failures.append("central console is missing context/readout/note panels")
    if data.get("chipCount", 0) < 2:
        failures.append("central console parameter transfer chips are missing")
    if data.get("stepCount", 0) < 3:
        failures.append("central console practice steps are missing")
    if not data.get("hasWorkbench"):
        failures.append("central console workbench is missing")
    if not data.get("hasNodeCanvas"):
        failures.append("central console is missing the draggable node canvas")
    if data.get("nodeCount", 0) < 5:
        failures.append("node canvas has too few draggable nodes")
    if data.get("edgeCount", 0) < 4:
        failures.append("node canvas has too few connecting edges")
    node_before = data.get("nodeBefore") or {}
    node_after = data.get("nodeAfter") or {}
    if abs(node_after.get("x", 0) - node_before.get("x", 0)) < 20:
        failures.append("node canvas first node did not move after pointer drag")
    if "平均相邻距离" not in data.get("nodeReadoutText", ""):
        failures.append("node canvas readout did not update with topology metrics")
    if not data.get("hasTrainingBus"):
        failures.append("central console is missing the training event bus")
    if data.get("eventLogCount", 0) < 2:
        failures.append("training event bus did not record control-change events")
    if data.get("eventSubscriberCount", 0) < 4:
        failures.append("training event bus subscribers are missing")
    for key in ("busLoss", "busGradient", "busFeature", "busNote"):
        if not data.get(key) or data.get(key) == "--":
            failures.append(f"training event bus subscriber did not update: {key}")
    if not data.get("labType"):
        failures.append("central console did not embed the native interactive lab")
    if data.get("labControlCount", 0) < 1:
        failures.append("central console lab has no controls")
    lab_rect = data.get("labRect") or {}
    if lab_rect.get("w", 0) <= 0 or lab_rect.get("h", 0) < 220:
        failures.append("central console lab is missing or collapsed")
    if len(data.get("labReadoutAfter", "")) < 12:
        failures.append("central console lab did not render a meaningful readout")
    if data.get("labReadoutBefore") == data.get("labReadoutAfter"):
        failures.append("central console lab did not respond to control changes")
    if data.get("consoleMetricCount", 0) < 3:
        failures.append("central console live metrics are missing")
    if len(data.get("consoleResultAfter", "")) < 40:
        failures.append("central console live result is too thin")
    if data.get("consoleResultBefore") == data.get("consoleResultAfter"):
        failures.append("central console live result did not update after changing controls")
    if len(data.get("noteAfter", "")) <= len(data.get("noteBefore", "")):
        failures.append("central console observation generator did not fill the note")
    if "我调了" not in data.get("noteAfter", "") or "这说明" not in data.get("noteAfter", ""):
        failures.append("central console generated note is missing observation structure")
    if not str(data.get("returnCourseHref", "")).startswith("#course/"):
        failures.append("central console is missing a return-course link")
    if data.get("returnHomeHref") != "#home":
        failures.append("central console is missing a return-home link")
    if data.get("revealCount", 0) < 4:
        failures.append("central console has too few motion reveal targets")
    if data.get("visibleRevealCount", 0) < 1:
        failures.append("central console reveal targets did not become visible")
    if not data.get("reducedMotion") and data.get("appAnimation") in {"", "none"}:
        failures.append("central console route entry animation is missing")
    if data.get("scrollWidth", 0) > data.get("clientWidth", 0) + 4:
        failures.append("central console page has horizontal overflow")
    return failures


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Static HTML UX smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=9230)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="deep_learning_book_edge_", ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen(
            [
                str(edge_path()),
                "--headless=new",
                f"--remote-debugging-port={args.port}",
                "--remote-allow-origins=*",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_cdp(args.port)
            reports = [inspect_page(args.base_url, page, args.port, args.screenshots) for page in PAGES]
        finally:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    failures = [f"{report['name']}: {failure}" for report in reports for failure in report["failures"]]
    if failures:
        print("\nUX check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nUX check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

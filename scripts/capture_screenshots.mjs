import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const port = 8879;
const baseURL = `http://127.0.0.1:${port}`;
const outDir = path.join(root, "docs", "screenshots");

function waitForServer() {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const tick = async () => {
      try {
        const response = await fetch(baseURL);
        if (response.ok) {
          resolve();
          return;
        }
      } catch (error) {
        // Server is still starting.
      }
      if (Date.now() - start > 20000) {
        reject(new Error("Timed out waiting for local screenshot server"));
        return;
      }
      setTimeout(tick, 250);
    };
    tick();
  });
}

async function capture(page, route, filename, action) {
  await page.goto(`${baseURL}/${route}`, { waitUntil: "networkidle" });
  if (action) await action(page);
  await page.screenshot({ path: path.join(outDir, filename), fullPage: false });
}

await mkdir(outDir, { recursive: true });

const server = spawn("python", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
  cwd: root,
  stdio: "ignore",
  windowsHide: true,
});

try {
  await waitForServer();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });

  await capture(page, "#home", "home.png");
  await capture(page, "#course/part4%2Ftransformer_models", "course-transformer.png", async (current) => {
    await current.getByTestId("toc-animation").click();
    await current.waitForTimeout(350);
  });
  await capture(page, "#console/part4%2Ftransformer_models", "console-builder.png", async (current) => {
    await current.getByTestId("model-preset").selectOption("transformer_mini");
    await current.getByTestId("load-preset").click();
    await current.getByTestId("export-code").click();
    await current.waitForTimeout(350);
  });
  await capture(page, "#console/part4%2Ftransformer_models", "shape-diagnostic.png", async (current) => {
    await current.getByTestId("model-preset").selectOption("shape_error");
    await current.getByTestId("load-preset").click();
    await current.waitForTimeout(350);
  });
  await capture(page, "#course/part7%2Finterview_quiz", "interview-camp.png");

  await browser.close();
} finally {
  server.kill();
}

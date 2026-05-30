import { expect, test } from "@playwright/test";

test.describe("course learning loop", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => localStorage.clear());
  });

  test("student can enter a lesson, follow the guide, mark progress, and continue", async ({ page }) => {
    await page.goto("/#course/part1%2Fmath_primer");

    await expect(page.getByTestId("course-layout")).toBeVisible();
    await expect(page.getByTestId("lesson-roadmap")).toBeVisible();
    await expect(page.getByTestId("lesson-three-minute")).toBeVisible();
    await expect(page.getByTestId("course-toc")).toBeVisible();
    await expect(page.getByTestId("learning-actions")).toBeVisible();

    await page.getByTestId("toc-animation").click();
    await expect(page.locator("#course-animation")).toBeInViewport();

    await page.getByTestId("toc-lab").click();
    await expect(page.locator("#course-lab")).toBeInViewport();

    await page.getByTestId("mark-understood").click();
    await expect(page.getByTestId("learning-status")).toContainText(/理解|understood|保存/);

    await page.getByTestId("mark-review").click();
    await expect(page.getByTestId("learning-status")).toContainText(/复习|review/);

    const progress = await page.evaluate(() => JSON.parse(localStorage.getItem("deep-learning-book-progress-v1")));
    expect(progress.understood).toContain("part1/math_primer");
    expect(progress.review).toContain("part1/math_primer");

    await expect(page.getByTestId("next-lesson")).toBeVisible();
  });

  test("beginner and advanced modes keep source code out of the beginner path", async ({ page }) => {
    await page.goto("/#course/part2%2F01_convolution_visual");

    await expect(page.getByTestId("developer-source")).toBeHidden();
    await page.getByTestId("mode-advanced").click();
    await expect(page.getByTestId("developer-source")).toBeVisible();
    await page.getByTestId("mode-beginner").click();
    await expect(page.getByTestId("developer-source")).toBeHidden();
  });

  test("tensor lesson shows rewritten source-marked article", async ({ page }) => {
    await page.goto("/#course/part1%2F01_tensors_gradients");

    await page.getByTestId("toc-reading").click();
    await expect(page.locator("[data-source-annotated-lesson]")).toBeVisible();
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("来源标注版");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("张量：先看 shape，再看数字");
    await expect(page.locator("[data-source-annotated-lesson] .source-marker").first()).toContainText("[S1]");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("PyTorch");
  });

  test("math primer shows rewritten source-marked article", async ({ page }) => {
    await page.goto("/#course/part1%2Fmath_primer");

    await page.getByTestId("toc-reading").click();
    await expect(page.locator("[data-source-annotated-lesson]")).toBeVisible();
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("为什么深度学习要先补这点数学");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("线性代数：看懂 shape");
    await expect(page.locator("[data-source-annotated-lesson] .source-marker").first()).toContainText("[S1]");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("梯度下降");
  });

  test("convolution lesson shows rewritten source-marked article", async ({ page }) => {
    await page.goto("/#course/part2%2F01_convolution_visual");

    await page.getByTestId("toc-reading").click();
    await expect(page.locator("[data-source-annotated-lesson]")).toBeVisible();
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("卷积到底在图像上找什么");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("stride 和 padding");
    await expect(page.locator("[data-source-annotated-lesson] .source-marker").first()).toContainText("[S1]");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("Conv2d");
  });

  test("rnn intuition lesson shows rewritten source-marked article", async ({ page }) => {
    await page.goto("/#course/part3%2F01_rnn_intuition");

    await page.getByTestId("toc-reading").click();
    await expect(page.locator("[data-source-annotated-lesson]")).toBeVisible();
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("RNN 到底记住了什么");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("隐藏状态");
    await expect(page.locator("[data-source-annotated-lesson] .source-marker").first()).toContainText("[S1]");
    await expect(page.locator("[data-source-annotated-lesson]")).toContainText("GRU");
  });
});

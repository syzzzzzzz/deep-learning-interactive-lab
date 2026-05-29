import { expect, test } from "@playwright/test";

test.describe("central console model builder", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.evaluate(() => localStorage.clear());
  });

  test("loads transformer preset, diagnoses shapes, exports code, and restores saved graph", async ({ page }) => {
    await page.goto("/#console/part4%2Ftransformer_models");

    await expect(page.getByTestId("console-builder")).toBeVisible();

    await page.getByTestId("model-preset").selectOption("transformer_mini");
    await page.getByTestId("load-preset").click();

    await expect(page.getByTestId("model-node-list")).toContainText("MultiHeadAttention");
    await expect(page.getByTestId("model-node-list")).toContainText("TransformerEncoder");
    await expect(page.getByTestId("shape-diagnostics")).toContainText("OK");

    await page.getByTestId("export-code").click();
    await expect(page.getByTestId("code-export")).toContainText("class VisualModel");
    await expect(page.getByTestId("code-export")).toContainText("nn.MultiheadAttention");

    await page.getByTestId("save-graph").click();
    const savedGraph = await page.getByTestId("graph-json").inputValue();
    expect(savedGraph).toContain("transformer_mini");
    expect(savedGraph).toContain("MultiHeadAttention");

    await page.getByTestId("clear-graph").click();
    await expect(page.getByTestId("model-node-list")).toContainText("No layers");

    await page.getByTestId("graph-json").fill(savedGraph);
    await page.getByTestId("load-graph").click();
    await expect(page.getByTestId("model-node-list")).toContainText("TransformerEncoder");

    await page.getByTestId("model-preset").selectOption("shape_error");
    await page.getByTestId("load-preset").click();
    await expect(page.getByTestId("shape-diagnostics")).toContainText("shape mismatch");
    await expect(page.getByTestId("shape-diagnostics")).toContainText("Insert Linear");
  });
});

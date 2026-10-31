// Guarda docs/img/simulador.png (claro) y simulador-dark.png (oscuro) con el reparto óptimo.
// Requiere la página servida en local: uv run python -m http.server -d site 8000
import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:8000/";
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL ?? "msedge" });
for (const mode of ["light", "dark"]) {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 2,
    colorScheme: mode,
  });
  await page.goto(url);
  await page.waitForSelector("#chart-roi svg");
  await page.click("#btn-optimal");
  const path = mode === "light" ? "docs/img/simulador.png" : "docs/img/simulador-dark.png";
  await page.locator("#simulador .sim").screenshot({ path });
  await page.close();
  console.log(`Captura guardada en ${path}`);
}
await browser.close();

// Guarda docs/img/simulador.png con el reparto óptimo aplicado.
// Requiere la página servida en local: uv run python -m http.server -d site 8000
import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:8000/";
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL ?? "msedge" });
const page = await browser.newPage({
  viewport: { width: 1280, height: 800 },
  deviceScaleFactor: 2,
  colorScheme: "light",
});
await page.goto(url);
await page.waitForSelector("#chart-roi svg");
await page.click("#btn-optimal");
await page.locator("#simulador").screenshot({ path: "docs/img/simulador.png" });
await browser.close();
console.log("Captura guardada en docs/img/simulador.png");

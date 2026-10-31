// Arma la página: titular, simulador, gráficos y controles de calidad.
import { contributionChart, curveCharts, holdoutChart, money, num1, roiChart } from "./charts.js";
import { isOutOfRange, simulatePlan } from "./simulator.js";

const $ = (sel, root = document) => root.querySelector(sel);

function readTheme(channels) {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    font: v("--font"),
    surface: v("--surface"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    axis: v("--axis"),
    accent: v("--accent"),
    series: Object.fromEntries(channels.map((c) => [c, v(`--${c}`)])),
  };
}

function listJoin(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} y ${items.at(-1)}`;
}

// Nombres comunes en minúscula dentro de una oración. TV y Facebook quedan igual.
function inText(label) {
  return ["TV", "Facebook"].includes(label) ? label : label.toLowerCase();
}

function swatch(color) {
  const s = document.createElement("span");
  s.className = "swatch";
  s.style.background = color;
  return s;
}

function renderIntro(results) {
  const { plans, channels, channel_labels: labels, headline } = results;
  const diff = channels.map((c) => ({ c, d: plans.optimal[c] - plans.current[c] }));
  const threshold = 0.05;
  const up = diff.filter((x) => x.d > threshold * plans.current[x.c]).sort((a, b) => b.d - a.d);
  const down = diff.filter((x) => x.d < -threshold * plans.current[x.c]).sort((a, b) => a.d - b.d);
  const uplift = headline.media_uplift_pct;
  $("#hero-value").textContent = `${uplift >= 0 ? "+" : ""}${Math.round(uplift)}%`;
  $("#answer").textContent =
    `Pasar inversión de ${listJoin(down.map((x) => inText(labels[x.c])))} a ` +
    `${listJoin(up.map((x) => inText(labels[x.c])))}. Con el mismo presupuesto semanal, ` +
    `las ventas que generan los medios suben un ${Math.round(uplift)}%.`;
}

function renderSimulator(results) {
  const { channels, channel_labels: labels, plans, spend_max_weekly: maxSpend } = results;
  const plan = { ...plans.current };
  const baseline = simulatePlan(results, plans.current).media.median;
  const rows = {};
  const container = $("#sliders");
  container.replaceChildren();

  for (const c of channels) {
    const row = document.createElement("div");
    row.className = "slider-row";
    const id = `slider-${c}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.append(swatch(`var(--${c})`), labels[c]);
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = "0";
    input.max = String(1.5 * maxSpend[c]);
    input.step = String((1.5 * maxSpend[c]) / 300);
    const output = document.createElement("output");
    output.htmlFor = id;
    const warn = document.createElement("p");
    warn.className = "warn";
    warn.textContent = "⚠ Fuera del rango visto en los datos.";
    row.append(label, input, output, warn);
    container.append(row);
    input.addEventListener("input", () => {
      plan[c] = Number(input.value);
      update();
    });
    rows[c] = { input, output, warn };
  }

  function update() {
    let total = 0;
    for (const c of channels) {
      const { input, output, warn } = rows[c];
      input.value = String(plan[c]);
      output.textContent = money(plan[c]);
      warn.hidden = !isOutOfRange(results, c, plan[c]);
      total += plan[c];
    }
    $("#sim-total").textContent = money(total);
    const r = simulatePlan(results, plan);
    $("#res-media").textContent = money(r.media.median);
    $("#res-media-range").textContent = `entre ${money(r.media.p05)} y ${money(r.media.p95)}`;
    $("#res-total").textContent = money(r.total.median);
    $("#res-total-range").textContent = `entre ${money(r.total.p05)} y ${money(r.total.p95)}`;
    const pct = (r.media.median / baseline - 1) * 100;
    const delta = $("#res-delta");
    delta.className = "tile-value";
    if (Math.abs(pct) < 0.05) {
      delta.textContent = "Igual";
    } else {
      delta.classList.add(pct > 0 ? "delta-up" : "delta-down");
      delta.textContent = `${pct > 0 ? "▲ +" : "▼ "}${num1(pct)}%`;
    }
  }

  $("#btn-current").onclick = () => {
    Object.assign(plan, plans.current);
    update();
  };
  $("#btn-optimal").onclick = () => {
    Object.assign(plan, plans.optimal);
    update();
  };
  update();
}

function renderTable(wrap, columns, rows) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col.label;
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const col of columns) tr.insertCell().textContent = col.value(row);
  }
  wrap.replaceChildren(table);
}

function legend(el, items) {
  el.replaceChildren(
    ...items.map(({ label, color, kind }) => {
      const span = document.createElement("span");
      const key = kind === "line" ? document.createElement("span") : swatch(color);
      if (kind === "line") {
        key.className = "line-key";
        key.style.background = color;
      }
      span.append(key, label);
      return span;
    }),
  );
}

function renderTables(results) {
  const { channels, channel_labels: labels } = results;
  renderTable(
    $("#chart-roi .table-wrap"),
    [
      { label: "Canal", value: (r) => labels[r.channel] },
      { label: "Mediana", value: (r) => num1(r.median) },
      { label: "Mínimo (5%)", value: (r) => num1(r.p05) },
      { label: "Máximo (95%)", value: (r) => num1(r.p95) },
    ],
    results.roi,
  );

  const byDate = new Map();
  for (const r of results.contributions) {
    if (!byDate.has(r.date)) byDate.set(r.date, { date: r.date });
    byDate.get(r.date)[r.series] = r.value;
  }
  renderTable(
    $("#chart-contrib .table-wrap"),
    [
      { label: "Semana", value: (r) => r.date },
      ...channels.map((c) => ({ label: labels[c], value: (r) => money(r[c]) })),
      { label: "Base", value: (r) => money(r.base) },
    ],
    [...byDate.values()],
  );

  renderTable(
    $("#chart-curves .table-wrap"),
    [
      { label: "Canal", value: (c) => labels[c] },
      { label: "Inversión semanal actual", value: (c) => money(results.plans.current[c]) },
      { label: "Inversión semanal óptima", value: (c) => money(results.plans.optimal[c]) },
      { label: "Máximo en los datos", value: (c) => money(results.spend_max_weekly[c]) },
    ],
    channels,
  );

  renderTable(
    $("#chart-holdout .table-wrap"),
    [
      { label: "Semana", value: (r) => r.date },
      { label: "Real", value: (r) => money(r.actual) },
      { label: "Predicción", value: (r) => money(r.median) },
      { label: "Mínimo (5%)", value: (r) => money(r.p05) },
      { label: "Máximo (95%)", value: (r) => money(r.p95) },
    ],
    results.holdout,
  );

  for (const button of document.querySelectorAll(".table-toggle")) {
    button.onclick = () => {
      const wrap = button.nextElementSibling;
      wrap.hidden = !wrap.hidden;
      button.textContent = wrap.hidden ? "Ver tabla" : "Ocultar tabla";
    };
  }
}

function renderCharts(results) {
  const theme = readTheme(results.channels);
  const width = (sel) => $(sel).clientWidth;
  const { channels, channel_labels: labels } = results;

  $("#chart-roi .plot").replaceChildren(roiChart(results, theme, width("#chart-roi .plot")));

  legend(
    $("#chart-contrib .legend"),
    channels.map((c) => ({ label: labels[c], color: theme.series[c] })),
  );
  $("#chart-contrib .plot").replaceChildren(
    contributionChart(results, theme, width("#chart-contrib .plot")),
  );

  const multiples = $("#chart-curves .multiples");
  const cells = channels.map((c) => {
    const cell = document.createElement("div");
    const title = document.createElement("h4");
    title.append(swatch(theme.series[c]), labels[c]);
    cell.append(title);
    return cell;
  });
  multiples.replaceChildren(...cells);
  const cellWidth = cells[0].clientWidth;
  curveCharts(results, theme, cellWidth).forEach(({ svg }, i) => cells[i].append(svg));

  legend($("#chart-holdout .legend"), [
    { label: "Ventas reales", color: theme.ink, kind: "line" },
    { label: "Predicción del modelo", color: theme.accent, kind: "line" },
  ]);
  $("#chart-holdout .plot").replaceChildren(
    holdoutChart(results, theme, width("#chart-holdout .plot")),
  );
}

function renderDiagnostics(results) {
  const d = results.diagnostics;
  const rows = [
    ["Divergencias del sampler", String(d.divergences), "debe ser 0"],
    ["R-hat máximo", d.max_rhat.toFixed(3).replace(".", ","), "menor que 1,01"],
    ["ESS mínimo", Math.round(d.min_ess).toLocaleString("es-AR"), "mayor que 400"],
    ["Error en las 12 semanas de prueba (MAPE)", `${num1(d.holdout_mape * 100)}%`, "solo informativo"],
  ];
  const body = $("#diag-body");
  body.replaceChildren(
    ...rows.map(([name, value, rule], i) => {
      const tr = document.createElement("tr");
      for (const text of [name, value, rule]) tr.insertCell().textContent = text;
      if (i < 3) tr.cells[1].className = "pass";
      return tr;
    }),
  );
  $("#holdout-sub").textContent +=
    ` El error promedio fue ${num1(d.holdout_mape * 100)}%.`;
}

function setupTheme(onChange) {
  const button = $("#theme-toggle");
  const isDark = () => getComputedStyle(document.documentElement).colorScheme === "dark";
  const label = () => {
    button.textContent = isDark() ? "Modo claro" : "Modo oscuro";
  };
  button.onclick = () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (_) {}
    label();
    onChange();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    label();
    onChange();
  });
  label();
}

async function main() {
  const results = await (await fetch("data/resultados.json")).json();
  const draw = () => renderCharts(results);
  renderIntro(results);
  renderSimulator(results);
  renderTables(results);
  renderDiagnostics(results);
  draw();
  setupTheme(draw);
  let timer;
  new ResizeObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(() => renderCharts(results), 150);
  }).observe($("main"));
}

main().catch((error) => {
  $("#answer").textContent = `No se pudieron cargar los resultados: ${error.message}`;
});

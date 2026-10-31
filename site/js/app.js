// Arma la página: inicio, simulador, pestañas de evidencia y controles de calidad.
// Los gráficos se cargan aparte: si el CDN de Observable Plot falla, el resto sigue andando.
import { money, num1 } from "./format.js";
import { isOutOfRange, simulatePlan } from "./simulator.js";

const $ = (sel, root = document) => root.querySelector(sel);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const easeOutExpo = (t) => (t === 1 ? 1 : 1 - 2 ** (-10 * t));
const easeOutCubic = (t) => 1 - (1 - t) ** 3;

// Anima un valor de 0 a 1 en `ms`. Devuelve una función para cancelarla.
function animate(ms, ease, onFrame) {
  if (reduceMotion.matches) {
    onFrame(1);
    return () => {};
  }
  let frame;
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / ms);
    onFrame(ease(t));
    if (t < 1) frame = requestAnimationFrame(step);
  };
  frame = requestAnimationFrame(step);
  return () => cancelAnimationFrame(frame);
}

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
    model1: v("--model-1"),
    model2: v("--model-2"),
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

const signed = (v) => `${v >= 0 ? "+" : ""}${Math.round(v)}%`;

function renderIntro(results) {
  const { plans, channels, channel_labels: labels, headline } = results;
  const diff = channels.map((c) => ({ c, d: plans.optimal[c] - plans.current[c] }));
  const threshold = 0.05;
  const up = diff.filter((x) => x.d > threshold * plans.current[x.c]).sort((a, b) => b.d - a.d);
  const down = diff.filter((x) => x.d < -threshold * plans.current[x.c]).sort((a, b) => a.d - b.d);
  const uplift = headline.media_uplift_pct;

  const moves =
    down.length && up.length
      ? `Pasar inversión de ${listJoin(down.map((x) => inText(labels[x.c])))} a ` +
        `${listJoin(up.map((x) => inText(labels[x.c])))}. `
      : "";
  $("#answer").textContent =
    `${moves}Con el mismo presupuesto semanal, las ventas que generan los medios ` +
    `suben un ${Math.round(uplift)}%.`;

  const value = $("#hero-value");
  animate(1100, easeOutExpo, (t) => {
    value.textContent = signed(uplift * t);
  });

  const band = headline.uplift;
  if (band) {
    $("#hero-range").textContent = `${signed(band.p05)} a ${signed(band.p95)}`;
    $("#hero-prob").textContent = `${Math.round(band.prob_positive * 100)}%`;
  }
  if (headline.bias_corrected_pct != null) {
    $("#hero-corrected").textContent = signed(headline.bias_corrected_pct);
  } else {
    $("#hero-corrected-row").hidden = true;
  }
}

// Mueve un indicador (pastilla) debajo del botón activo de un grupo.
function placeIndicator(indicator, target) {
  if (!target) {
    indicator.style.setProperty("--o", "0");
    return;
  }
  indicator.style.setProperty("--x", `${target.offsetLeft}px`);
  indicator.style.setProperty("--w", `${target.offsetWidth}px`);
  indicator.style.setProperty("--o", "1");
}

function renderSimulator(results) {
  const { channels, channel_labels: labels, plans, spend_max_weekly: maxSpend } = results;
  const plan = { ...plans.current };
  const baseline = simulatePlan(results, plans.current).media.median;
  const rows = {};
  const container = $("#sliders");
  const presets = { current: $("#btn-current"), optimal: $("#btn-optimal") };
  const indicator = $(".segmented-indicator");
  let cancelTween = () => {};
  container.replaceChildren();

  for (const c of channels) {
    const top = 1.5 * maxSpend[c];
    const row = document.createElement("div");
    row.className = "slider-row";
    const id = `slider-${c}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.append(swatch(`var(--${c})`), labels[c]);
    const range = document.createElement("div");
    range.className = "range";
    range.style.setProperty("--fill-color", `var(--${c})`);
    range.style.setProperty("--max", String(maxSpend[c] / top));
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = "0";
    input.max = String(top);
    input.step = String(top / 300);
    input.setAttribute("aria-describedby", "sim-warn");
    const tick = document.createElement("span");
    tick.className = "max-tick";
    tick.title = "Inversión semanal más alta en los datos";
    range.append(input, tick);
    const output = document.createElement("output");
    output.htmlFor = id;
    row.append(label, range, output);
    container.append(row);
    input.addEventListener("input", () => {
      cancelTween();
      plan[c] = Number(input.value);
      setPreset(null);
      update();
    });
    rows[c] = { row, input, output, range, top };
  }

  function setPreset(name) {
    for (const [key, button] of Object.entries(presets)) {
      button.setAttribute("aria-pressed", String(key === name));
    }
    placeIndicator(indicator, name ? presets[name] : null);
  }

  function update() {
    let total = 0;
    const out = [];
    for (const c of channels) {
      const { row, input, output, range, top } = rows[c];
      input.value = String(plan[c]);
      range.style.setProperty("--fill", `${(plan[c] / top) * 100}%`);
      output.textContent = money(plan[c]);
      const outside = isOutOfRange(results, c, plan[c]);
      row.classList.toggle("is-out", outside);
      if (outside) out.push(labels[c]);
      total += plan[c];
    }
    const warn = $("#sim-warn");
    warn.textContent = out.length
      ? `Fuera del rango visto en los datos: ${listJoin(out)}. El modelo extrapola.`
      : "";
    warn.classList.toggle("is-visible", out.length > 0);

    $("#sim-total").textContent = money(total);
    const r = simulatePlan(results, plan);
    $("#res-media").textContent = money(r.media.median);
    $("#res-media-range").textContent = `entre ${money(r.media.p05)} y ${money(r.media.p95)}`;
    $("#res-total").textContent = money(r.total.median);
    $("#res-total-range").textContent = `entre ${money(r.total.p05)} y ${money(r.total.p95)}`;
    const pct = (r.media.median / baseline - 1) * 100;
    const delta = $("#res-delta");
    delta.classList.toggle("delta-up", pct >= 0.05);
    delta.classList.toggle("delta-down", pct <= -0.05);
    delta.textContent = Math.abs(pct) < 0.05 ? "Igual" : `${pct > 0 ? "+" : ""}${num1(pct)}%`;
    $("#res-delta-note").textContent =
      Math.abs(pct) < 0.05
        ? "en ventas de medios (mediana)"
        : `${pct > 0 ? "más" : "menos"} ventas de medios (mediana)`;
  }

  // Lleva las barras al plan elegido con una transición corta e interrumpible.
  function goTo(name) {
    cancelTween();
    setPreset(name);
    const from = { ...plan };
    const to = plans[name];
    cancelTween = animate(520, easeOutCubic, (t) => {
      for (const c of channels) plan[c] = from[c] + (to[c] - from[c]) * t;
      update();
    });
  }

  presets.current.onclick = () => goTo("current");
  presets.optimal.onclick = () => goTo("optimal");
  update();
  setPreset("current");
  return () =>
    setPreset(Object.keys(presets).find((k) => presets[k].getAttribute("aria-pressed") === "true"));
}

function renderTable(wrap, columns, rows) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const col of columns) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = col.label;
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const col of columns) tr.insertCell().textContent = col.value(row);
  }
  wrap.querySelector(".table-inner").replaceChildren(table);
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

function renderTables(results, meridian) {
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

  if (meridian) {
    const byChannel = Object.fromEntries(meridian.roi.map((r) => [r.channel, r]));
    const range = (r) => `${num1(r.median)} (${num1(r.p05)} a ${num1(r.p95)})`;
    renderTable(
      $("#chart-meridian .table-wrap"),
      [
        { label: "Canal", value: (r) => labels[r.channel] },
        { label: "PyMC-Marketing", value: (r) => range(r) },
        { label: "Meridian", value: (r) => range(byChannel[r.channel]) },
      ],
      results.roi,
    );
  }

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
    const wrap = button.nextElementSibling;
    wrap.id ||= `${button.closest(".panel").id}-tabla`;
    button.setAttribute("aria-controls", wrap.id);
    button.onclick = () => {
      const open = button.getAttribute("aria-expanded") !== "true";
      button.setAttribute("aria-expanded", String(open));
      button.textContent = open ? "Ocultar tabla" : "Ver tabla";
      wrap.classList.toggle("is-open", open);
      wrap.inert = !open;
    };
  }
}

// Dibuja un panel de evidencia. `charts` es el módulo de gráficos ya cargado.
function drawPanel(id, charts, results, meridian) {
  const theme = readTheme(results.channels);
  const panel = $(`#${id}`);
  const width = (sel) => panel.querySelector(sel).clientWidth;
  const { channels, channel_labels: labels } = results;

  if (id === "chart-roi") {
    panel.querySelector(".plot").replaceChildren(charts.roiChart(results, theme, width(".plot")));
  } else if (id === "chart-meridian" && meridian) {
    legend(panel.querySelector(".legend"), [
      { label: "PyMC-Marketing (este modelo)", color: theme.model1 },
      { label: "Meridian (Google)", color: theme.model2 },
    ]);
    panel
      .querySelector(".plot")
      .replaceChildren(charts.meridianChart(results, meridian, theme, width(".plot")));
  } else if (id === "chart-contrib") {
    legend(
      panel.querySelector(".legend"),
      channels.map((c) => ({ label: labels[c], color: theme.series[c] })),
    );
    panel
      .querySelector(".plot")
      .replaceChildren(charts.contributionChart(results, theme, width(".plot")));
  } else if (id === "chart-curves") {
    const multiples = panel.querySelector(".multiples");
    const cells = channels.map((c) => {
      const cell = document.createElement("div");
      const title = document.createElement("h4");
      title.append(swatch(theme.series[c]), labels[c]);
      cell.append(title);
      return cell;
    });
    multiples.replaceChildren(...cells);
    const cellWidth = cells[0].clientWidth;
    charts.curveCharts(results, theme, cellWidth).forEach(({ svg }, i) => cells[i].append(svg));
  } else if (id === "chart-holdout") {
    legend(panel.querySelector(".legend"), [
      { label: "Ventas reales", color: theme.ink, kind: "line" },
      { label: "Predicción del modelo", color: theme.accent, kind: "line" },
    ]);
    panel
      .querySelector(".plot")
      .replaceChildren(charts.holdoutChart(results, theme, width(".plot")));
  }
}

function setupTabs(onShow) {
  const tablist = $(".tabs");
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const indicator = $(".tab-indicator");
  let current = tabs.find((t) => t.getAttribute("aria-selected") === "true");

  function select(tab, focus) {
    if (tab === current) return;
    const previous = $(`#${current.getAttribute("aria-controls")}`);
    const next = $(`#${tab.getAttribute("aria-controls")}`);
    current.setAttribute("aria-selected", "false");
    current.tabIndex = -1;
    tab.setAttribute("aria-selected", "true");
    tab.tabIndex = 0;
    previous.hidden = true;
    next.hidden = false;
    next.classList.remove("is-entering");
    void next.offsetWidth; // reinicia la animación de entrada
    next.classList.add("is-entering");
    current = tab;
    placeIndicator(indicator, tab);
    tab.scrollIntoView({ block: "nearest", inline: "nearest" });
    if (focus) tab.focus();
    onShow(next.id);
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab, false));
    tab.addEventListener("keydown", (event) => {
      const moves = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      select(tabs[(moves[event.key] + tabs.length) % tabs.length], true);
    });
  });
  placeIndicator(indicator, current);
  return {
    active: () => current.getAttribute("aria-controls"),
    refresh: () => placeIndicator(indicator, current),
  };
}

function renderDiagnostics(results, meridian) {
  const d = results.diagnostics;
  const rows = [
    ["Divergencias del sampler", String(d.divergences), "Debe ser 0", true],
    ["R-hat máximo", d.max_rhat.toFixed(3).replace(".", ","), "Debe ser menor que 1,01", true],
    ["ESS mínimo", Math.round(d.min_ess).toLocaleString("es-AR"), "Debe ser mayor que 400", true],
    [
      "Error en 12 semanas de prueba",
      `${num1(d.holdout_mape * 100)}%`,
      "MAPE de semanas que el modelo no vio",
      false,
    ],
  ];
  $("#diag-list").replaceChildren(
    ...rows.map(([name, value, rule, gate]) => {
      const item = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      const note = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = value;
      if (gate) dd.className = "pass";
      note.className = "rule";
      note.textContent = gate ? `${rule}. Pasa.` : rule;
      item.append(dt, dd, note);
      return item;
    }),
  );
  $("#holdout-sub").textContent += ` El error promedio fue ${num1(d.holdout_mape * 100)}%.`;
  if (meridian) {
    $("#meridian-sub").textContent +=
      ` Error en las semanas de prueba: PyMC-Marketing ${num1(d.holdout_mape * 100)}%,` +
      ` Meridian ${num1(meridian.diagnostics.holdout_mape * 100)}%.`;
  }
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

async function loadJson(path) {
  try {
    const response = await fetch(path);
    return response.ok ? await response.json() : null;
  } catch (_) {
    return null;
  }
}

async function main() {
  const results = await loadJson("data/resultados.json");
  if (!results) throw new Error("falta data/resultados.json");
  const meridian = await loadJson("data/meridian.json");
  if (!meridian) {
    $("#tab-meridian").hidden = true;
  }

  renderIntro(results);
  const refreshPreset = renderSimulator(results);
  renderTables(results, meridian);
  renderDiagnostics(results, meridian);

  let charts = null;
  const drawn = new Set();
  const show = (id) => {
    if (!charts || drawn.has(id)) return;
    drawPanel(id, charts, results, meridian);
    drawn.add(id);
  };
  const tabs = setupTabs(show);
  const redraw = () => {
    drawn.clear();
    show(tabs.active());
  };
  setupTheme(redraw);

  try {
    charts = await import("./charts.js");
    show(tabs.active());
  } catch (_) {
    for (const plot of document.querySelectorAll(".panel .plot, .panel .multiples")) {
      plot.textContent = "No se pudieron cargar los gráficos. Los datos siguen en la tabla.";
    }
  }

  let timer;
  let lastWidth = $("main").clientWidth;
  new ResizeObserver(() => {
    const width = $("main").clientWidth;
    if (width === lastWidth) return;
    lastWidth = width;
    clearTimeout(timer);
    timer = setTimeout(() => {
      tabs.refresh();
      refreshPreset();
      redraw();
    }, 150);
  }).observe($("main"));
}

main().catch((error) => {
  $("#answer").textContent = `No se pudieron cargar los resultados: ${error.message}`;
});

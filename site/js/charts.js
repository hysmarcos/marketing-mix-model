// Gráficos con Observable Plot. Cada función devuelve un nodo SVG listo para insertar.
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";
import { band, saturationCurve, weeklyContribution } from "./simulator.js";

const short = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const dateFmt = new Intl.DateTimeFormat("es-AR", { day: "numeric", month: "short", year: "numeric" });

export function money(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `$ ${short.format(v / 1e6)} M`;
  if (a >= 1e3) return `$ ${short.format(v / 1e3)} k`;
  return `$ ${short.format(v)}`;
}
export const num1 = (v) => decimal.format(v);
export const day = (d) => dateFmt.format(d instanceof Date ? d : new Date(d));

function baseStyle(theme) {
  return {
    background: "transparent",
    color: theme.muted,
    fontFamily: theme.font,
    fontSize: "12px",
    overflow: "visible",
  };
}

function tipFormat(theme) {
  return { fill: theme.surface, stroke: theme.axis, textPadding: 10 };
}

export function roiChart(results, theme, width) {
  const data = [...results.roi]
    .sort((a, b) => b.median - a.median)
    .map((r) => ({ ...r, label: results.channel_labels[r.channel] }));
  return Plot.plot({
    width,
    height: 56 + data.length * 44,
    marginLeft: 96,
    marginRight: 48,
    style: baseStyle(theme),
    x: { label: "Ventas por cada peso invertido →", grid: true, nice: true },
    y: { domain: data.map((d) => d.label), label: null, tickSize: 0 },
    marks: [
      Plot.ruleX([1], { stroke: theme.axis }),
      Plot.ruleY(data, {
        y: "label",
        x1: "p05",
        x2: "p95",
        stroke: (d) => theme.series[d.channel],
        strokeWidth: 2,
        strokeLinecap: "round",
      }),
      Plot.dot(data, {
        y: "label",
        x: "median",
        r: 5,
        fill: (d) => theme.series[d.channel],
        stroke: theme.surface,
        strokeWidth: 2,
      }),
      Plot.text(data, {
        y: "label",
        x: "p95",
        text: (d) => num1(d.median),
        dx: 8,
        textAnchor: "start",
        fill: theme.ink,
      }),
      Plot.tip(
        data,
        Plot.pointerY({
          y: "label",
          x: "median",
          ...tipFormat(theme),
          title: (d) =>
            `${d.label}\nMediana: ${num1(d.median)}\nRango del 90%: ${num1(d.p05)} a ${num1(d.p95)}`,
        }),
      ),
    ],
  });
}

const monthFmt = new Intl.DateTimeFormat("es-AR", { month: "long", year: "numeric" });

// Suma los aportes semanales por mes. Las semanas sueltas son muy ruidosas para leer.
export function monthlyContributions(results) {
  const months = new Map();
  for (const r of results.contributions) {
    if (r.series === "base") continue;
    const key = r.date.slice(0, 7);
    if (!months.has(key)) months.set(key, { month: new Date(`${key}-01T00:00:00`), total: 0 });
    const entry = months.get(key);
    entry[r.series] = (entry[r.series] ?? 0) + r.value;
    entry.total += r.value;
  }
  return [...months.values()];
}

export function contributionChart(results, theme, width) {
  const { channels, channel_labels: labels } = results;
  const months = monthlyContributions(results);
  const rows = months.flatMap((m) =>
    channels.map((c) => ({ month: m.month, series: c, value: m[c] })),
  );
  return Plot.plot({
    width,
    height: 320,
    marginLeft: 56,
    style: baseStyle(theme),
    x: { label: null },
    y: { label: "Ventas por mes", grid: true, tickFormat: money },
    marks: [
      Plot.rectY(
        rows,
        Plot.stackY({
          x: "month",
          interval: "month",
          y: "value",
          order: channels,
          fill: (d) => theme.series[d.series],
          stroke: theme.surface,
          strokeWidth: 1,
          insetLeft: 1,
          insetRight: 1,
        }),
      ),
      Plot.ruleY([0], { stroke: theme.axis }),
      Plot.tip(
        months,
        Plot.pointerX({
          x: "month",
          y: "total",
          ...tipFormat(theme),
          title: (d) =>
            [
              monthFmt.format(d.month),
              ...channels.map((c) => `${labels[c]}: ${money(d[c])}`),
              `Total medios: ${money(d.total)}`,
            ].join("\n"),
        }),
      ),
    ],
  });
}

export function curveCharts(results, theme, width) {
  const { channels, channel_labels: labels, plans, draws, scales } = results;
  const curves = Object.fromEntries(channels.map((c) => [c, saturationCurve(results, c, 60)]));
  const yMax = Math.max(...channels.flatMap((c) => curves[c].map((p) => p.p95)));
  return channels.map((c) => {
    const spend = plans.current[c];
    const point = {
      spend,
      ...band(weeklyContribution(spend, draws.lam[c], draws.beta[c], scales.channel[c], scales.target)),
    };
    const svg = Plot.plot({
      width,
      height: 170,
      marginLeft: 48,
      marginBottom: 32,
      style: baseStyle(theme),
      x: { label: "Inversión semanal", tickFormat: money, ticks: 2 },
      y: { label: null, domain: [0, yMax], tickFormat: money, ticks: 3, grid: true },
      marks: [
        Plot.areaY(curves[c], { x: "spend", y1: "p05", y2: "p95", fill: theme.series[c], fillOpacity: 0.12 }),
        Plot.lineY(curves[c], { x: "spend", y: "median", stroke: theme.series[c], strokeWidth: 2 }),
        Plot.ruleY([0], { stroke: theme.axis }),
        Plot.dot([point], {
          x: "spend",
          y: "median",
          r: 5,
          fill: theme.series[c],
          stroke: theme.surface,
          strokeWidth: 2,
        }),
        Plot.tip(
          curves[c],
          Plot.pointerX({
            x: "spend",
            y: "median",
            ...tipFormat(theme),
            title: (d) =>
              `Inversión: ${money(d.spend)}\nVentas: ${money(d.median)}\nRango del 90%: ${money(d.p05)} a ${money(d.p95)}`,
          }),
        ),
      ],
    });
    return { channel: c, label: labels[c], svg };
  });
}

export function holdoutChart(results, theme, width) {
  const rows = results.holdout.map((d) => ({ ...d, date: new Date(d.date) }));
  return Plot.plot({
    width,
    height: 300,
    marginLeft: 56,
    style: baseStyle(theme),
    x: { label: null },
    y: { label: "Ventas por semana", grid: true, tickFormat: money },
    marks: [
      Plot.areaY(rows, { x: "date", y1: "p05", y2: "p95", fill: theme.accent, fillOpacity: 0.12 }),
      Plot.lineY(rows, { x: "date", y: "median", stroke: theme.accent, strokeWidth: 2 }),
      Plot.lineY(rows, { x: "date", y: "actual", stroke: theme.ink, strokeWidth: 2 }),
      Plot.dot(rows, { x: "date", y: "actual", r: 4, fill: theme.ink, stroke: theme.surface, strokeWidth: 2 }),
      Plot.ruleX(rows, Plot.pointerX({ x: "date", stroke: theme.ink2 })),
      Plot.tip(
        rows,
        Plot.pointerX({
          x: "date",
          y: "actual",
          ...tipFormat(theme),
          title: (d) =>
            `${day(d.date)}\nReal: ${money(d.actual)}\nPredicción: ${money(d.median)}\nRango del 90%: ${money(d.p05)} a ${money(d.p95)}`,
        }),
      ),
    ],
  });
}

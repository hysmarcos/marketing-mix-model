import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  isOutOfRange,
  percentile,
  saturationCurve,
  simulatePlan,
  weeklyContribution,
} from "./simulator.js";

const fixture = JSON.parse(
  readFileSync(new URL("../../tests/fixtures/simulator_cases.json", import.meta.url), "utf-8"),
);
const results = JSON.parse(
  readFileSync(new URL("../data/resultados.json", import.meta.url), "utf-8"),
);

const close = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));

test("paridad con Python en todos los casos", () => {
  assert.equal(fixture.cases.length, 4);
  for (const c of fixture.cases) {
    const r = simulatePlan(fixture.results_subset, c.plan);
    for (const k of ["media", "total"]) {
      for (const q of ["median", "p05", "p95"]) {
        assert.ok(close(r[k][q], c.expected[k][q]), `${k}.${q}: ${r[k][q]} vs ${c.expected[k][q]}`);
      }
    }
  }
});

test("inversión 0 da aporte 0 sin NaN", () => {
  assert.deepEqual(weeklyContribution(0, [1, 2], [1, 1], 10, 100), [0, 0]);
});

test("aporte conocido: z = 1", () => {
  const [v] = weeklyContribution(5, [2], [0.5], 10, 100);
  assert.ok(Math.abs(v - 100 * 0.5 * 0.46211715726000974) < 1e-9);
});

test("percentile igual a numpy", () => {
  assert.ok(Math.abs(percentile([1, 2, 3, 4], 5) - 1.15) < 1e-12);
  assert.equal(percentile([4, 1, 3, 2], 50), 2.5);
});

test("isOutOfRange", () => {
  const max = results.spend_max_weekly.tv_S;
  assert.equal(isOutOfRange(results, "tv_S", max * 1.01), true);
  assert.equal(isOutOfRange(results, "tv_S", max), false);
});

test("curva de saturación crece y empieza en 0", () => {
  const curve = saturationCurve(results, "facebook_S", 20);
  assert.equal(curve.length, 20);
  assert.equal(curve[0].spend, 0);
  assert.equal(curve[0].median, 0);
  for (let i = 1; i < curve.length; i++) assert.ok(curve[i].median >= curve[i - 1].median);
  assert.ok(curve.every((p) => p.p05 <= p.median && p.median <= p.p95));
});

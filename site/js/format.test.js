import { test } from "node:test";
import assert from "node:assert/strict";

process.env.TZ = "America/Argentina/Buenos_Aires";
const { day, month } = await import("./format.js");

test("la fecha de la semana no se corre un día en Argentina", () => {
  assert.match(day("2019-08-26"), /^26 /);
});

test("el mes no se corre en Argentina", () => {
  assert.match(month("2019-08-01"), /^agosto/);
});

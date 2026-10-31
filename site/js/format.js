// Formato de montos y fechas para la página. Sin dependencias, para poder probarlo en Node.
// Las fechas del JSON son días sin hora. Se leen y se muestran en UTC para que no se corran.

const short = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat("es-AR", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const dateFmt = new Intl.DateTimeFormat("es-AR", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const monthFmt = new Intl.DateTimeFormat("es-AR", { month: "long", year: "numeric", timeZone: "UTC" });

export function money(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `$ ${short.format(v / 1e6)} M`;
  if (a >= 1e3) return `$ ${short.format(v / 1e3)} k`;
  return `$ ${short.format(v)}`;
}

export const num1 = (v) => decimal.format(v);
export const day = (d) => dateFmt.format(d instanceof Date ? d : new Date(d));
export const month = (d) => monthFmt.format(d instanceof Date ? d : new Date(d));

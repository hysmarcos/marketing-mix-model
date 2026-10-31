// Cálculo del simulador. Misma fórmula que src/mmm/export.py. No toca el DOM.

export function logistic(z) {
  return (1 - Math.exp(-z)) / (1 + Math.exp(-z));
}

// Igual que numpy.percentile con interpolación lineal.
export function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
}

export function band(values) {
  return {
    median: percentile(values, 50),
    p05: percentile(values, 5),
    p95: percentile(values, 95),
  };
}

// Aporte semanal en estado estable, un valor por muestra del modelo.
export function weeklyContribution(spend, lam, beta, channelScale, targetScale) {
  return lam.map((l, i) => targetScale * beta[i] * logistic((l * spend) / channelScale));
}

// Pesos del adstock geométrico normalizado: alpha^l / suma, para l = 0..lMax-1.
export function adstockWeights(alpha, lMax) {
  const w = Array.from({ length: lMax }, (_, l) => alpha ** l);
  const sum = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / sum);
}

// Inversión con adstock en cada semana futura. history: últimas lMax-1 semanas reales,
// de la más vieja a la más nueva. Desde la primera semana futura se invierte spend.
export function adstockedSpend(alpha, history, spend, weeks) {
  const lMax = history.length + 1;
  const w = adstockWeights(alpha, lMax);
  const series = [...history, ...new Array(weeks).fill(spend)];
  return Array.from({ length: weeks }, (_, t) => {
    let x = 0;
    for (let lag = 0; lag < lMax; lag++) x += w[lag] * series[history.length + t - lag];
    return x;
  });
}

// Ventas de medios en todo el horizonte, una por muestra. Misma fórmula que export.py.
export function mediaDraws(results, plan) {
  const { draws, scales, channels, history } = results;
  const weeks = results.meta.horizon_weeks;
  const n = draws.baseline_weekly.length;
  const media = new Array(n).fill(0);
  for (const c of channels) {
    for (let i = 0; i < n; i++) {
      const x = adstockedSpend(draws.alpha[c][i], history[c], plan[c], weeks);
      for (const xt of x) {
        media[i] += scales.target * draws.beta[c][i] * logistic((draws.lam[c][i] * xt) / scales.channel[c]);
      }
    }
  }
  return media;
}

export function simulatePlan(results, plan) {
  const weeks = results.meta.horizon_weeks;
  const media = mediaDraws(results, plan);
  const total = media.map((v, i) => v + weeks * results.draws.baseline_weekly[i]);
  return { media: band(media), total: band(total) };
}

export function saturationCurve(results, channel, points = 50) {
  const max = 1.5 * results.spend_max_weekly[channel];
  const { draws, scales } = results;
  return Array.from({ length: points }, (_, k) => {
    const spend = (max * k) / (points - 1);
    const values = weeklyContribution(
      spend,
      draws.lam[channel],
      draws.beta[channel],
      scales.channel[channel],
      scales.target,
    );
    return { spend, ...band(values) };
  });
}

export function isOutOfRange(results, channel, spend) {
  return spend > results.spend_max_weekly[channel];
}

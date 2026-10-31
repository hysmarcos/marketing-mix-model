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

export function simulatePlan(results, plan) {
  const { draws, scales, channels } = results;
  const weeks = results.meta.horizon_weeks;
  const media = new Array(draws.baseline_weekly.length).fill(0);
  for (const c of channels) {
    const contribution = weeklyContribution(
      plan[c],
      draws.lam[c],
      draws.beta[c],
      scales.channel[c],
      scales.target,
    );
    contribution.forEach((v, i) => {
      media[i] += v;
    });
  }
  const mediaTotal = media.map((v) => weeks * v);
  const total = mediaTotal.map((v, i) => v + weeks * draws.baseline_weekly[i]);
  return { media: band(mediaTotal), total: band(total) };
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

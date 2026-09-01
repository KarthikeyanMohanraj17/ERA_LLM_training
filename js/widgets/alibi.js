// ALiBi. Two things worth seeing, side by side:
//   left  — what the mechanism does: subtract m·distance from the raw score,
//           with a real score axis, so the penalty is a quantity not a vibe.
//   right — what that does to the actual attention weights after softmax:
//           the built-in recency bias, made concrete.
// The "evaluate past training length" toggle adds a learned/sinusoidal position
// signal for contrast — same axis, because both end up as a pre-softmax score.

function initAlibi(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const trainedLen = 16;
  const RAW = 2.0;      // the same raw relevance at every distance, held fixed
  let extendPast = false;
  let slope = 0.35;     // per-head penalty slope m

  // Fixed pseudo-noise: erratic-looking but stable across redraws.
  const noise = (x) => Math.sin(x * 12.9) * Math.sin(x * 3.7 + 1.3);

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const maxLen = extendPast ? trainedLen * 2.5 : trainedLen;
    const xt = [];
    for (let d = 0; d <= maxLen; d += extendPast ? 8 : 4) xt.push(d);

    // the penalty is linear forever, so let the axis follow it instead of
    // clamping — a floor would look like ALiBi stops penalising
    const yMin = Math.min(-6, Math.floor((RAW - slope * maxLen) / 2) * 2);
    const yTicks = [];
    const tStep = Math.ceil(Math.abs(yMin - 3) / 5 / 2) * 2;
    for (let v = 3; v >= yMin; v -= tStep) yTicks.push(v);

    const split = Math.max(0.55, 1 - 300 / w);
    const ch = chart(ctx, {
      w, h,
      pad: { l: 58, r: w * (1 - split) + 34, t: 42, b: 76 },
      x: { min: 0, max: maxLen, ticks: xt, fmt: (v) => String(Math.round(v)),
           label: "distance from the query (tokens)" },
      y: { min: yMin, max: 3, ticks: yTicks, fmt: (v) => v.toFixed(0),
           label: "pre-softmax attention score" },
      grid: true,
    });

    // raw score — identical relevance at every distance, so any difference
    // below is the mechanism, not the content
    ctx.strokeStyle = cssVar("--text-faint");
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ch.l, ch.py(RAW));
    ctx.lineTo(ch.r, ch.py(RAW));
    ctx.stroke();
    ctx.setLineDash([]);

    // ALiBi: raw - m*d, a straight line defined at every distance
    ctx.strokeStyle = cssVar("--good");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let d = 0; d <= maxLen; d += 0.2) {
      const y = ch.py(RAW - slope * d);
      if (d === 0) ctx.moveTo(ch.px(d), y); else ctx.lineTo(ch.px(d), y);
    }
    ctx.stroke();

    // trained-length boundary + a learned position signal for contrast
    if (extendPast) {
      const bx = ch.px(trainedLen);
      ctx.strokeStyle = cssVar("--border-strong");
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(bx, ch.t); ctx.lineTo(bx, ch.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = font(10);
      ctx.fillStyle = cssVar("--text-faint");
      ctx.textAlign = "center";
      ctx.fillText("trained up to here", bx, ch.t - 8);

      ctx.strokeStyle = cssVar("--bad");
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let d = 0; d <= maxLen; d += 0.2) {
        const over = Math.max(0, d - trainedLen);
        // in-distribution: some learned, non-linear recency shape that works
        const learned = RAW - 1.7 * Math.log1p(d) + 0.3 * Math.sin(d * 0.9);
        const v = learned + noise(d) * Math.min(3.5, over * 0.7);
        const y = ch.py(Math.max(yMin, Math.min(3, v)));
        if (d === 0) ctx.moveTo(ch.px(d), y); else ctx.lineTo(ch.px(d), y);
      }
      ctx.stroke();
    }

    // ---------- right: what it does to the actual attention weights ----------
    const dists = [0, 1, 2, 3, 4, 5, 6, 7];
    const weights = softmax(dists.map((d) => RAW - slope * d));
    const ch2 = chart(ctx, {
      w, h,
      pad: { l: w * split + 26, r: 22, t: 42, b: 76 },
      x: { min: -0.5, max: 7.5, ticks: [0, 2, 4, 6], fmt: (v) => String(Math.round(v)),
           label: "distance from the query" },
      y: { min: 0, max: Math.max(0.4, Math.ceil(weights[0] * 10) / 10), ticks: 5, fmt: (v) => v.toFixed(2),
           label: "softmax weight" },
      grid: true,
    });
    const bw = (ch2.plotW / 8) * 0.6;
    dists.forEach((d, i) => {
      const x = ch2.px(d), y = ch2.py(weights[i]);
      ctx.fillStyle = cssVar("--accent");
      ctx.globalAlpha = 0.85;
      roundRect(ctx, x - bw / 2, y, bw, ch2.b - y, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
    });
    ctx.textAlign = "right";
    ctx.font = font(10.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("identical raw scores in — this shape is pure penalty", ch2.r, ch2.t - 12);

    // ---------- header + legend ----------
    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`penalty = m × distance,  m = ${slope.toFixed(2)}   ·   at 10 tokens apart: −${(slope * 10).toFixed(1)}`, 20, 22);

    legend(ctx, 20, h - 34, [
      { color: cssVar("--text-faint"), label: `raw score in (${RAW.toFixed(1)} everywhere)`, shape: "line" },
      { color: cssVar("--good"), label: "ALiBi: raw − m·d", shape: "line" },
    ].concat(extendPast ? [{ color: cssVar("--bad"), label: "a learned position signal", shape: "line" }] : []), 10.5, w - 20);

    caption(ctx, 20, h - 12,
      extendPast
        ? "The green line is a formula, so it keeps its exact shape past the boundary. The red one never learned those distances."
        : "Flip the toggle to evaluate past the length the model was trained on.");
  }

  bindToggle(root, ".ab-extend", "evaluated past training length", "evaluated within training length", (v) => {
    extendPast = v;
    draw();
  }, false);

  bindSlider(root, ".ab-slope", ".ab-slope-out", (v) => { slope = v; draw(); }, { fmt: (v) => v.toFixed(2) });

  window.addEventListener("resize", draw);
  draw();
}

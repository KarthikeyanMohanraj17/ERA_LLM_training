// DDDGDDDG-style depth schedule: D = fixed-state (Gated DeltaNet-style) layer,
// G = sparse-attention layer. Only G layers carry a KV cache, so the ratio is a
// direct dial between memory cost and how often the model can read a specific
// earlier token.
//
// HONESTY NOTE. An earlier version of this widget drew two bars interpolated
// between "LightningLM 0.1V's two reported points: 1 G/8 -> 8 G/8 is 8.0x cache,
// ~1.41x compute". Those numbers are not in arXiv:2606.07404. Grepping the
// paper's full text: "1.41" 0 hits, "8.0x" 0 hits, "KV cache" 0 hits. Table 5
// fixes the D:G ratio at 3:1 for every stage it trains (6:2 at 8 layers, 15:5 at
// 20) and never sweeps it, so neither endpoint exists as a measurement.
//
// What is left is arithmetic anyone can check and nobody needs to have measured:
// if only G layers hold a KV cache, cache is proportional to the G count. That
// is the bar below, labelled as this page's own derivation, not as a result.

function initDepthSchedule(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let gCount = 2; // out of 8

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const n = 8;
    const cell = Math.min((w - 60) / n, 62);
    const x0 = 30;
    const y0 = 34;

    const pattern = Array(n).fill("D");
    if (gCount > 0) {
      const step = n / gCount;
      for (let i = 0; i < gCount; i++) pattern[Math.min(n - 1, Math.round(i * step + step - 1))] = "G";
    }

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText("one repeating 8-layer block, bottom of the stack → top", x0, 20);

    pattern.forEach((t, i) => {
      const x = x0 + i * cell;
      ctx.fillStyle = t === "G" ? cssVar("--accent-2") : cssVar("--accent");
      ctx.globalAlpha = t === "G" ? 0.9 : 0.5;
      roundRect(ctx, x + 3, y0, cell - 6, 38, 6);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0d12";
      ctx.font = font(14, 700);
      ctx.textAlign = "center";
      ctx.fillText(t, x + cell / 2, y0 + 25);
    });
    legend(ctx, x0, y0 + 58, [
      { color: cssVar("--accent"), label: "D — fixed-state layer, no KV cache at all", shape: "box" },
      { color: cssVar("--accent-2"), label: "G — sparse attention, this is what costs cache", shape: "box" },
    ], 10.5, w - 20);

    // Derived, not measured: only G layers cache, so cache is proportional to
    // the G count. Shown against the 3:1 schedule the paper actually trains.
    const PAPER_G = 2; // the paper's own 8-layer config is 6 D : 2 G
    const cacheShare = gCount / n;
    const vsPaper = gCount / PAPER_G;

    const ch = chart(ctx, {
      w, h,
      pad: { l: 210, r: 76, t: h - 106, b: 52 },
      x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => Math.round(v * 100) + "%",
           label: "share of layers holding a KV cache  (= G layers ÷ 8)" },
      y: { min: -0.5, max: 1.5, ticks: [] },
      grid: false, gridX: true,
    });
    [
      { label: `this schedule (${gCount} G)`, v: cacheShare, color: cssVar("--accent-2"), txt: `${Math.round(cacheShare * 100)}%` },
      { label: "paper's own 6D:2G", v: PAPER_G / n, color: cssVar("--border-strong"), txt: "25%" },
    ].forEach((r, i) => {
      const y = ch.py(1 - i) - 9;
      ctx.fillStyle = r.color;
      ctx.fillRect(ch.px(0), y, Math.max(2, ch.px(r.v) - ch.px(0)), 18);
      ctx.textAlign = "right";
      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(r.label, ch.l - 10, y + 13);
      ctx.textAlign = "left";
      ctx.font = font(11, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.fillText(r.txt, ch.px(r.v) + 8, y + 13);
    });
    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = gCount === PAPER_G ? cssVar("--good") : cssVar("--accent");
    ctx.fillText(
      gCount === PAPER_G
        ? "this is the schedule the paper actually trains (3 D per G, held constant at every scale)"
        : `${vsPaper.toFixed(2)}× the KV cache of the paper's 3:1 schedule — a config it never trained`,
      x0, y0 + 106);

    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`${gCount} G layer${gCount === 1 ? "" : "s"} per 8  ·  ${n - gCount} D layers`, x0, y0 + 84);
    caption(ctx, x0, h - 8, "Cache share is arithmetic, not a measurement: arXiv:2606.07404 fixes D:G at 3:1 everywhere (6:2 at 8 layers, 15:5 at 20) and never sweeps it.");
  }

  bindSlider(root, ".ds-g", ".ds-g-out", (v) => { gCount = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  window.addEventListener("resize", draw);
  draw();
}

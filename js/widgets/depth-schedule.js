// DDDGDDDG-style depth schedule: D = fixed-state layer, G = sparse-attention layer.
// Cache and compute multipliers are linearly interpolated between the two reported
// V4 data points (1 G/8 -> 8 G/8: cache 8.0x, compute ~1.41x) — labeled as such,
// not presented as a measured curve at every point.

function initDepthSchedule(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let gCount = 2; // out of 8

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const n = 8;
    const cell = Math.min((w - 60) / n, 60);
    const x0 = (w - cell * n) / 2;
    const y0 = 30;

    // build a schedule with gCount G's spread as evenly as possible
    const pattern = Array(n).fill("D");
    if (gCount > 0) {
      const step = n / gCount;
      for (let i = 0; i < gCount; i++) pattern[Math.min(n - 1, Math.round(i * step + step - 1))] = "G";
    }

    pattern.forEach((t, i) => {
      const x = x0 + i * cell;
      ctx.fillStyle = t === "G" ? cssVar("--accent-2") : cssVar("--accent");
      ctx.globalAlpha = t === "G" ? 0.9 : 0.55;
      roundRect(ctx, x + 3, y0, cell - 6, 40, 6);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0d12";
      ctx.font = "13px var(--mono), monospace";
      ctx.textAlign = "center";
      ctx.fillText(t, x + cell / 2, y0 + 25);
    });

    // interpolate between the two reported data points
    const frac = (gCount - 1) / (8 - 1); // 0 at gCount=1, 1 at gCount=8
    const cacheMult = 1 + frac * (8.0 - 1);
    const computeMult = 1 + frac * (1.41 - 1);

    ctx.font = "12px var(--mono), monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`${gCount} sparse-attention (G) layer(s) per 8, ${8 - gCount} fixed-state (D) layers`, x0, y0 + 70);
    ctx.fillStyle = cssVar("--bad");
    ctx.fillText(`KV-cache multiplier (interpolated): ${cacheMult.toFixed(2)}×`, x0, y0 + 94);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`mixing-compute multiplier (interpolated): ${computeMult.toFixed(2)}×`, x0, y0 + 114);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("anchored to LightningLM 0.1V's two reported points: 1/8 G → 8/8 G is 8.0× cache, ~1.41× compute", x0, y0 + 140);
  }

  bindSlider(root, ".ds-g", ".ds-g-out", (v) => { gCount = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  window.addEventListener("resize", draw);
  draw();
}

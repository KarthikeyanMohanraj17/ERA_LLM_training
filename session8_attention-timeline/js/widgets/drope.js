// DroPE: illustrative "quality vs. position" curves, with and without dropping
// the positional embeddings and recalibrating. Not a measured benchmark — it
// shows the shape of the paper's claim (restores in-distribution quality,
// extends the usable range), the same way the RoPE-scaling widgets do.
// Both curves are drawn at once so the comparison doesn't depend on memory.

function initDrope(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const trainedLen = 16; // "8K", in arbitrary units
  let ext = 3;           // how far past the trained length to look
  let dropeOn = true;

  const noise = (d) => Math.sin(d * 9.1) * 0.06;

  function qualityWithout(d) {
    if (d <= trainedLen) return 0.9;
    return Math.max(0.08, 0.9 - (d - trainedLen) * 0.045 + noise(d));
  }
  function qualityWith(d) {
    // recalibration costs a hair of peak quality near the boundary, then holds
    if (d <= trainedLen * 0.9) return 0.9;
    if (d <= trainedLen * 1.15) return 0.84;
    return 0.87;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const maxLen = trainedLen * ext;
    const xt = [];
    for (let i = 0; i <= ext; i++) xt.push(trainedLen * i);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 74, r: 26, t: 54, b: 78 },
      x: { min: 0, max: maxLen, ticks: xt, fmt: (v) => (v === 0 ? "0" : (v / trainedLen) + "×"),
           label: "evaluation context length, relative to what the model was trained on" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2),
           label: "illustrative quality" },
      grid: true,
    });

    const bx = ch.px(trainedLen);
    ctx.strokeStyle = cssVar("--border-strong");
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx, ch.t); ctx.lineTo(bx, ch.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = font(10);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "center";
    ctx.fillText("original trained length", bx, ch.t - 8);

    function curve(fn, color, active) {
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.globalAlpha = active ? 1 : 0.4;
      ctx.beginPath();
      for (let d = 0; d <= maxLen; d += maxLen / 240) {
        const x = ch.px(d), y = ch.py(fn(d));
        if (d === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // both curves always drawn; the toggle just decides which one is in focus
    curve(qualityWithout, cssVar("--bad"), !dropeOn);
    curve(qualityWith, cssVar("--good"), dropeOn);

    legend(ctx, ch.l, ch.b + 44, [
      { color: cssVar("--good"), label: "with DroPE — positions dropped, then recalibrated", shape: "line" },
      { color: cssVar("--bad"), label: "without — RoPE evaluated past what it ever saw", shape: "line" },
    ], 10.5, w - 20);

    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = dropeOn ? cssVar("--good") : cssVar("--bad");
    const at = maxLen;
    ctx.fillText(
      `at ${ext}× the trained length: with DroPE ${qualityWith(at).toFixed(2)}  ·  without ${qualityWithout(at).toFixed(2)}`,
      ch.l, 24
    );
    caption(ctx, ch.l, h - 10, "Illustrative shape, not measured perplexity — the paper's own numbers are in the source link above.");
  }

  bindToggle(root, ".dp-toggle", "DroPE: on", "DroPE: off", (v) => { dropeOn = v; draw(); }, true);
  bindSlider(root, ".dp-ext", ".dp-ext-out", (v) => { ext = v; draw(); }, { fmt: (v) => v + "×" });

  window.addEventListener("resize", draw);
  draw();
}

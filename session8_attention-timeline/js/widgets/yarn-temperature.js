// YaRN's second contribution, and the one that distinguishes it from bloc97's
// NTK-aware post: the attention temperature fix.
//
// Interpolation compresses positions, which pulls rotation angles closer
// together, which pulls the pre-softmax scores closer together. Softmax over
// closer scores is FLATTER — higher entropy, less decisive attention — even
// though nothing about the content changed. YaRN divides the scores by a
// temperature t < 1 before softmax to put the sharpness back.
//
// Every number here is computed live from the scores shown. Nothing is asserted.

function initYarnTemperature(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const RAW = [4.0, 2.0, 1.0, 0.5, 0, 0, 0, 0]; // matches this section's worked example
  const SQUEEZE = 0.4;                           // what 8x interpolation does to the spread
  let t = 1.0;                                   // temperature; 1.0 = no correction

  const entropy = (p) => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);

    const original = softmax(RAW);
    const corrected = softmax(RAW.map((v) => (v * SQUEEZE) / t));
    const hOrig = entropy(original), hCorr = entropy(corrected);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 68, r: 26, t: 62, b: 96 },
      x: { min: -0.5, max: RAW.length - 0.5, ticks: [0, 2, 4, 6], fmt: (v) => String(Math.round(v)),
           label: "the eight keys this query is scoring, best first" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2),
           label: "softmax weight" },
      grid: true,
    });

    const bw = (ch.plotW / RAW.length) * 0.6;
    RAW.forEach((_, i) => {
      const x = ch.px(i);
      // the pre-interpolation distribution, as a hollow reference outline
      ctx.strokeStyle = cssVar("--text-faint");
      ctx.setLineDash([3, 2]);
      ctx.lineWidth = 1;
      ctx.strokeRect(x - bw / 2, ch.py(original[i]), bw, ch.b - ch.py(original[i]));
      ctx.setLineDash([]);
      // what the model actually gets after interpolation + this temperature
      const y = ch.py(corrected[i]);
      ctx.fillStyle = cssVar("--accent");
      ctx.globalAlpha = 0.85;
      roundRect(ctx, x - bw / 2, y, bw, ch.b - y, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (i < 4) {
        ctx.font = font(10.5, 700);
        ctx.fillStyle = cssVar("--text");
        ctx.textAlign = "center";
        ctx.fillText(corrected[i].toFixed(2), x, y - 6);
      }
    });

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`raw scores [${RAW.slice(0, 4).join(", ")}, 0, 0, 0, 0]  ·  8× interpolation multiplies the spread by ${SQUEEZE}  ·  then ÷ t before softmax`, 20, 22);

    const exact = Math.abs(t - SQUEEZE) < 0.001;
    ctx.font = font(12, 700);
    ctx.fillStyle = exact ? cssVar("--good") : t < 0.99 ? cssVar("--accent-2") : cssVar("--bad");
    ctx.fillText(
      exact
        ? `t = ${SQUEEZE} exactly cancels the squeeze — the distribution is bit-for-bit the original again`
        : t > 0.99
          ? `no correction (t = 1.00): attention is flatter than it was, purely from compressing positions`
          : `t = ${t.toFixed(2)}: partway back toward the original sharpness`,
      20, 44);

    legend(ctx, ch.l, ch.b + 44, [
      { color: cssVar("--accent"), label: "after interpolation, at this temperature", shape: "box" },
      { color: cssVar("--text-faint"), label: "before extension (the target)", shape: "box" },
    ], 10.5, w - 20);

    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`top weight ${corrected[0].toFixed(3)}   (was ${original[0].toFixed(3)})`, ch.l, ch.b + 68);
    ctx.fillStyle = hCorr > hOrig + 0.01 ? cssVar("--bad") : cssVar("--good");
    ctx.fillText(`entropy ${hCorr.toFixed(3)} nats   (was ${hOrig.toFixed(3)})`, ch.l + Math.min(320, ch.plotW * 0.5), ch.b + 68);
    caption(ctx, ch.l, h - 10,
      "Higher entropy means the model is spreading attention more thinly across the same eight keys — hedging, without having learned anything new.");
  }

  bindSlider(root, ".yt-temp", ".yt-temp-out", (v) => { t = v; draw(); }, { fmt: (v) => v.toFixed(2) });
  window.addEventListener("resize", draw);
  draw();
}

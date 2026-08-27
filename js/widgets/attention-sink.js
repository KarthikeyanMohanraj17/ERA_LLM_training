// Attention sinks / StreamingLLM: a rolling window KV cache with or without a
// few permanently-kept "sink" tokens at the very start of the sequence.

function initAttentionSink(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 28;
  const windowSize = 8;
  const sinkCount = 4;
  let position = 20;
  let keepSinks = true;

  function stability(pos) {
    // illustrative only: without sinks, quality degrades once the window has
    // slid past the start; with sinks, it stays flat. Matches the paper's
    // qualitative finding, not a measured perplexity curve.
    if (keepSinks) return 0.92;
    const slidPast = Math.max(0, pos - windowSize);
    return Math.max(0.15, 0.92 - slidPast * 0.035);
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const pad = 24;
    const cell = (w - pad * 2) / n;
    const y = 60;

    for (let j = 0; j < n; j++) {
      const x = pad + j * cell + cell / 2;
      const isSink = j < sinkCount;
      const inWindow = position - j >= 0 && position - j < windowSize;
      const evicted = !inWindow && !(keepSinks && isSink);
      ctx.beginPath();
      ctx.arc(x, y, j === position ? 8 : 5, 0, Math.PI * 2);
      if (j === position) {
        ctx.fillStyle = cssVar("--accent-2");
        ctx.fill();
      } else if (keepSinks && isSink) {
        ctx.fillStyle = cssVar("--good");
        ctx.fill();
      } else if (inWindow) {
        ctx.fillStyle = cssVar("--accent");
        ctx.fill();
      } else {
        ctx.strokeStyle = cssVar("--bad");
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    ctx.font = "11px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    ctx.fillText("green = kept sink tokens   blue = sliding window   dim ring = evicted", pad, y + 24);

    // quality-over-time curve
    const padL = pad, padT = 100, plotW = w - pad * 2, plotH = h - padT - 24;
    ctx.strokeStyle = cssVar("--border");
    ctx.strokeRect(padL, padT, plotW, plotH);
    ctx.strokeStyle = keepSinks ? cssVar("--good") : cssVar("--bad");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let p = 0; p <= n; p++) {
      const q = stability(p);
      const x = padL + (p / n) * plotW;
      const yv = padT + plotH - q * plotH;
      if (p === 0) ctx.moveTo(x, yv); else ctx.lineTo(x, yv);
    }
    ctx.stroke();
    ctx.font = "10.5px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("illustrative generation quality over a long stream (not measured perplexity)", padL, padT - 6);
  }

  bindSlider(root, ".as-pos", ".as-pos-out", (v) => { position = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  bindToggle(root, ".as-sinks", "sinks: kept", "sinks: evicted", (v) => {
    keepSinks = v;
    draw();
  }, true);

  window.addEventListener("resize", draw);
  draw();
}

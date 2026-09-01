// DeltaNet's actual contribution: not new correction math (that's the delta
// rule, two sections earlier) but a chunk-parallel training algorithm for the
// same recurrence. What changes is the number of sequential steps, nothing else.

function initDeltaParallel(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const T = 24; // illustrative sequence length, small enough that boxes stay legible
  let chunk = 1;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const x0 = 30, x1 = w - 30;
    const boxW = (x1 - x0) / T;
    const boxH = 24;
    const rowSeqY = 42, rowParY = 112;
    const steps = Math.ceil(T / chunk);

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("sequential (2021 recurrence) — each token waits for the one before it", x0, rowSeqY - 10);
    for (let i = 0; i < T; i++) {
      ctx.fillStyle = cssVar("--bad");
      ctx.globalAlpha = 0.7;
      roundRect(ctx, x0 + i * boxW + 1, rowSeqY, boxW - 2, boxH, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--bad");
    ctx.fillText(`${T} sequential steps`, x0, rowSeqY + boxH + 16);

    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("chunk-parallel (DeltaNet, 2024) — tokens inside one chunk computed together", x0, rowParY - 10);
    for (let c = 0; c < steps; c++) {
      const startI = c * chunk;
      const len = Math.min(chunk, T - startI);
      ctx.fillStyle = cssVar("--good");
      ctx.globalAlpha = 0.7;
      roundRect(ctx, x0 + startI * boxW + 1, rowParY, len * boxW - 2, boxH, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (chunk > 1) {
        ctx.strokeStyle = cssVar("--bg");
        ctx.lineWidth = 1;
        for (let i = startI + 1; i < startI + len; i++) {
          const xi = x0 + i * boxW;
          ctx.beginPath(); ctx.moveTo(xi, rowParY + 3); ctx.lineTo(xi, rowParY + boxH - 3); ctx.stroke();
        }
      }
    }
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`${steps} sequential step${steps === 1 ? "" : "s"} — each covers ${chunk} token${chunk === 1 ? "" : "s"} at once`, x0, rowParY + boxH + 16);

    tokenAxis(ctx, { x0, x1, y: rowParY + boxH + 30, n: T, step: 4, label: "token index within the training sequence" });

    ctx.textAlign = "left";
    ctx.font = font(12, 700);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`speed-up in sequential depth: ${T} / ${steps} = ${(T / steps).toFixed(1)}×`, x0, h - 28);
    caption(ctx, x0, h - 10,
      "Same correction per token, same result — only the count of one-after-another steps changes.");
  }

  bindSlider(root, ".dnp-chunk", ".dnp-chunk-out", (v) => { chunk = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  window.addEventListener("resize", draw);
  draw();
}

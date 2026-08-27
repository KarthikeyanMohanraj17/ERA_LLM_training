// Top-k sparse attention over compressed block summaries. Reused for both NSA
// and DSA — DSA's "lightning indexer" selection is the same core idea NSA proposed.

function initTopK(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const nBlocks = 16;
  // fixed pseudo-relevance scores so the ranking is stable across redraws
  const scores = [0.9, 0.2, 0.6, 0.15, 0.8, 0.35, 0.95, 0.1, 0.5, 0.75, 0.05, 0.65, 0.3, 0.85, 0.25, 0.4];
  let k = opts.defaultK || 4;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const pad = 24;
    const cell = (w - pad * 2) / nBlocks;

    const ranked = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
    const keepSet = new Set(ranked.slice(0, k).map((r) => r.i));

    for (let i = 0; i < nBlocks; i++) {
      const x = pad + i * cell;
      const kept = keepSet.has(i);
      const barH = scores[i] * 90;
      ctx.fillStyle = kept ? cssVar("--accent") : cssVar("--border-strong");
      ctx.globalAlpha = kept ? 0.9 : 0.5;
      ctx.fillRect(x + 2, 120 - barH, cell - 4, barH);
      ctx.globalAlpha = 1;
      if (kept) {
        ctx.strokeStyle = cssVar("--accent-2");
        ctx.strokeRect(x + 1, 120 - barH - 1, cell - 2, barH + 2);
      }
    }

    ctx.font = "11px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    ctx.fillText("block relevance score (from the low-rank indexer)", pad, 20);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`top-${k} blocks selected for expensive attention (outlined) — ${k}/${nBlocks} = ${Math.round((k / nBlocks) * 100)}% of blocks read`, pad, 148);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText("every block is still scored cheaply by the indexer — only the winners get full attention", pad, 168);
  }

  bindSlider(root, ".tk-k", ".tk-k-out", (v) => { k = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  window.addEventListener("resize", draw);
  draw();
}

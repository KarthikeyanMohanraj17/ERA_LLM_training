// Block-sparse attention: score every block cheaply, then pay full attention
// only where it is worth it.
//   opts.branches = true  -> NSA, which runs three branches in parallel
//   otherwise             -> DSA, the same select-top-k idea as one production path

function initTopK(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const nBlocks = 16;
  const WINDOW_BLOCKS = 2; // NSA's local branch
  // Fixed pseudo-relevance scores, so the ranking is stable across redraws.
  const scores = [0.9, 0.2, 0.6, 0.15, 0.8, 0.35, 0.95, 0.1, 0.5, 0.75, 0.05, 0.65, 0.3, 0.85, 0.25, 0.4];
  const branches = !!opts.branches;
  const UNIT = opts.unit || "block";           // NSA selects blocks, DSA selects tokens
  const UNITS = UNIT + "s";
  const SCALE = opts.unitScale || 1;           // DSA's real k is 2,048 tokens, not 3
  let k = opts.defaultK || 4;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);

    const ranked = scores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
    const keep = new Set(ranked.slice(0, k).map((r) => r.i));

    const ch = chart(ctx, {
      w, h,
      pad: { l: branches ? 150 : 70, r: 24, t: 34, b: branches ? 148 : 88 },
      x: { min: -0.5, max: nBlocks - 0.5, ticks: [0, 4, 8, 12, 15], fmt: (v) => String(Math.round(v)),
           label: UNIT === "token" ? "candidate tokens, sampled across the context" : "block index — the sequence chopped into fixed-size blocks of tokens" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2),
           label: "cheap relevance score" },
      grid: true,
    });

    const bw = (ch.plotW / nBlocks) * 0.66;
    const cutoff = ranked[k - 1].s;
    // the top-k threshold, as a line you can see the bars cross
    ctx.strokeStyle = cssVar("--accent-2");
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ch.l, ch.py(cutoff)); ctx.lineTo(ch.r, ch.py(cutoff)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = font(10);
    ctx.textAlign = "right";
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`top-${k} cutoff = ${cutoff.toFixed(2)}`, ch.r - 4, ch.py(cutoff) - 6);

    for (let i = 0; i < nBlocks; i++) {
      const kept = keep.has(i);
      const x = ch.px(i), y = ch.py(scores[i]);
      ctx.fillStyle = kept ? cssVar("--accent") : cssVar("--border-strong");
      ctx.globalAlpha = kept ? 0.9 : 0.45;
      roundRect(ctx, x - bw / 2, y, bw, ch.b - y, 3);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    legend(ctx, ch.l, ch.b + 48, [
      { color: cssVar("--accent"), label: `selected ${UNITS} — full, expensive attention`, shape: "box" },
      { color: cssVar("--border-strong"), label: "scored by the indexer, then skipped", shape: "box" },
    ], 10.5, w - 20);

    if (branches) {
      // NSA runs three branches at once; each reads a different slice.
      const rows = [
        { label: "compressed", hit: () => true, color: cssVar("--good") },
        { label: "selected (top-k)", hit: (i) => keep.has(i), color: cssVar("--accent") },
        { label: "sliding window", hit: (i) => i >= nBlocks - WINDOW_BLOCKS, color: cssVar("--accent-2") },
      ];
      rows.forEach((r, ri) => {
        const y = ch.b + 70 + ri * 22;
        ctx.textAlign = "right";
        ctx.font = font(10.5);
        ctx.fillStyle = cssVar("--text-dim");
        ctx.fillText(r.label, ch.l - 10, y + 8);
        for (let i = 0; i < nBlocks; i++) {
          const on = r.hit(i);
          ctx.fillStyle = on ? r.color : cssVar("--bg-elevated");
          ctx.globalAlpha = on ? (ri === 0 ? 0.45 : 0.9) : 1;
          ctx.fillRect(ch.px(i) - bw / 2, y, bw, 11);
          ctx.globalAlpha = 1;
        }
      });
      caption(ctx, ch.l, h - 8,
        `The three branches are summed. Full attention would read all ${nBlocks} blocks in detail; here only ${k} are.`);
    } else {
      ctx.textAlign = "left";
      ctx.font = font(12, 700);
      ctx.fillStyle = cssVar("--accent");
      ctx.fillText(`${k} of ${nBlocks} ${UNITS} read in full = ${Math.round((k / nBlocks) * 100)}% of the attention cost`, ch.l, h - 32);
      caption(ctx, ch.l, h - 12, SCALE > 1
        ? "Scaled down for legibility: V3.2-Exp keeps k = 2,048 individual tokens, so at 128K context it reads under 2% of them. Every token is still scored — the saving is in what gets READ, not what gets ranked."
        : "Every unit is still scored by the cheap indexer — the saving is in what gets READ, not what gets ranked.");
    }
  }

  bindSlider(root, ".tk-k", ".tk-k-out", (v) => { k = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  window.addEventListener("resize", draw);
  draw();
}

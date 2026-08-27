// Standard scaled dot-product attention: 6-token toy sequence, causal mask toggle,
// pick a query token and see its scores -> softmax weights -> weighted sum.

function initQKV(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const tokens = ["The", "cat", "sat", "on", "the", "mat"];
  // Fixed toy key/query dot-product scores (hand-picked so "sat" attends strongly to "cat").
  const rawScores = [
    [4, 1, 0, 0, 0, 0],
    [1, 4, 1, 0, 0, 0],
    [1, 5, 4, 1, 0, 0],
    [0, 1, 2, 4, 1, 0],
    [0, 0, 1, 2, 4, 2],
    [0, 0, 1, 1, 3, 4],
  ];
  let query = 2;
  let masked = true;

  function weightsFor(q) {
    const row = rawScores[q].slice(0, masked ? q + 1 : tokens.length);
    const scaled = row.map((v) => v / Math.sqrt(4));
    const w = softmax(scaled);
    while (w.length < tokens.length) w.push(0);
    return w;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const n = tokens.length;
    const pad = 40;
    const cell = Math.min((w - pad * 2) / n, 44);
    const gridW = cell * n;
    const x0 = (w - gridW) / 2;
    const y0 = 50;
    ctx.font = "12px var(--mono), monospace";
    ctx.textAlign = "center";

    // column labels (keys)
    ctx.fillStyle = cssVar("--text-faint");
    tokens.forEach((t, j) => {
      ctx.fillText(t, x0 + j * cell + cell / 2, y0 - 12);
    });

    const weights = weightsFor(query);

    for (let j = 0; j < n; j++) {
      const blocked = masked && j > query;
      const wgt = weights[j];
      const x = x0 + j * cell;
      const y = y0 + query * 0 ; // single row display
      const cx = x, cy = y0;
      ctx.save();
      if (blocked) {
        ctx.fillStyle = cssVar("--border");
        roundRect(ctx, cx + 2, cy + 2, cell - 4, cell - 4, 6);
        ctx.fill();
        ctx.strokeStyle = cssVar("--border-strong");
        ctx.beginPath();
        ctx.moveTo(cx + 6, cy + 6);
        ctx.lineTo(cx + cell - 6, cy + cell - 6);
        ctx.stroke();
      } else {
        const alpha = 0.15 + wgt * 0.8;
        ctx.fillStyle = `rgba(110,168,255,${alpha})`;
        roundRect(ctx, cx + 2, cy + 2, cell - 4, cell - 4, 6);
        ctx.fill();
        ctx.strokeStyle = cssVar("--accent");
        ctx.globalAlpha = 0.5;
        roundRect(ctx, cx + 2, cy + 2, cell - 4, cell - 4, 6);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = wgt > 0.35 ? "#0b0d12" : cssVar("--text-dim");
        ctx.fillText(wgt.toFixed(2), cx + cell / 2, cy + cell / 2 + 4);
      }
      ctx.restore();
    }

    // query marker row label
    ctx.fillStyle = cssVar("--accent-2");
    ctx.textAlign = "left";
    ctx.fillText(`query = "${tokens[query]}" (token ${query})`, x0, y0 + cell + 28);

    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText(masked ? "causal mask ON — future tokens blocked" : "causal mask OFF — full sequence visible", x0, y0 + cell + 48);
  }

  const qInput = bindSlider(root, ".qkv-query", ".qkv-query-out", (v) => {
    query = Math.round(v);
    draw();
  }, { fmt: (v) => tokens[Math.round(v)] });

  bindToggle(root, ".qkv-mask", "mask: ON", "mask: OFF", (v) => {
    masked = v;
    if (masked && query < 0) query = 0;
    draw();
  }, true);

  window.addEventListener("resize", draw);
  draw();
}

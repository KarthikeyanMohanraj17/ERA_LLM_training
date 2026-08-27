// Sparse Transformer attention patterns: full vs strided vs local vs their union.

function initSparsePattern(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 16;
  let mode = "union"; // full | strided | local | union
  const stride = 4;
  const localWindow = 3;

  function allowed(i, j) {
    if (j > i) return false; // causal
    if (mode === "full") return true;
    const strided = (i - j) % stride === 0;
    const local = i - j <= localWindow;
    if (mode === "strided") return strided;
    if (mode === "local") return local;
    return strided || local;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const pad = 24;
    const size = Math.min(w - pad * 2, h - pad * 2);
    const cell = size / n;
    const x0 = (w - size) / 2;
    const y0 = pad;

    let total = 0, kept = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (j > i) continue;
        total++;
        const on = allowed(i, j);
        if (on) kept++;
        const x = x0 + j * cell;
        const y = y0 + i * cell;
        ctx.fillStyle = on ? cssVar("--accent") : cssVar("--bg-elevated");
        ctx.globalAlpha = on ? 0.75 : 1;
        ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        ctx.globalAlpha = 1;
      }
    }
    ctx.strokeStyle = cssVar("--border");
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, size, size);

    ctx.font = "12px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-dim");
    ctx.textAlign = "left";
    ctx.fillText(`mode: ${mode}  —  ${kept} / ${total} query-key pairs kept (${Math.round((kept / total) * 100)}%)`, x0, y0 + size + 22);
  }

  root.querySelectorAll(".sp-mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      mode = btn.dataset.mode;
      root.querySelectorAll(".sp-mode").forEach((b) => b.classList.toggle("on", b === btn));
      draw();
    });
  });
  root.querySelector(`.sp-mode[data-mode="${mode}"]`)?.classList.add("on");

  window.addEventListener("resize", draw);
  draw();
}

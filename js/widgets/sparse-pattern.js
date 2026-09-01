// Sparse Transformer attention patterns as an attention matrix:
// row i = the query doing the looking, column j = the key being looked at.
// A filled cell means "this pair actually gets a dot product computed".

function initSparsePattern(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 16;
  let mode = "union"; // full | strided | local | union
  const stride = 4;
  const localWindow = 3;

  function allowed(i, j) {
    if (j > i) return false; // causal: a query never sees the future
    if (mode === "full") return true;
    const isStrided = (i - j) % stride === 0;
    const isLocal = i - j <= localWindow;
    if (mode === "strided") return isStrided;
    if (mode === "local") return isLocal;
    return isStrided || isLocal;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 74, r: Math.max(22, w - 74 - (h - 46 - 92)), t: 46, b: 92 },
      x: { min: -0.5, max: n - 0.5, ticks: [0, 4, 8, 12, 15], fmt: (v) => String(Math.round(v)),
           label: "key token j  —  the token being looked at" },
      // reversed range so row 0 sits at the top, the way an attention matrix is drawn
      y: { min: n - 0.5, max: -0.5, ticks: [0, 4, 8, 12, 15], fmt: (v) => String(Math.round(v)),
           label: "query token i  —  the token doing the looking" },
      grid: false,
    });

    const cell = ch.plotW / n;
    let causalPairs = 0, computed = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = ch.px(j - 0.5);
        const y = ch.py(i - 0.5);
        if (j > i) {
          // masked by causality: left empty, so it reads as "not applicable"
          // rather than "the pattern chose to skip this"
          continue;
        }
        causalPairs++;
        const on = allowed(i, j);
        if (on) computed++;
        ctx.fillStyle = on ? cssVar("--accent") : cssVar("--bg-elevated");
        ctx.globalAlpha = on ? 0.85 : 1;
        ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        ctx.globalAlpha = 1;
        if (!on) {
          ctx.strokeStyle = cssVar("--border");
          ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
      }
    }

    legend(ctx, ch.l, ch.b + 44, [
      { color: cssVar("--accent"), label: "dot product computed", shape: "box" },
      { color: cssVar("--bg-elevated"), label: "legal pair, skipped by the pattern", shape: "box" },
      { color: cssVar("--bg"), label: "blank = future token, masked by causality", shape: "box" },
    ], 10.5, w - 20);

    ctx.textAlign = "left";
    ctx.font = font(12);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(
      `mode: ${mode}  —  ${computed} of ${causalPairs} legal pairs computed (${Math.round((computed / causalPairs) * 100)}%)`,
      ch.l, ch.b + 66
    );
    caption(ctx, ch.l, h - 8,
      mode === "full"
        ? `every legal pair — this is the O(n²) cost the paper is trying to avoid`
        : `strided = every ${stride}th token back · local = the last ${localWindow} tokens · union = both`);
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

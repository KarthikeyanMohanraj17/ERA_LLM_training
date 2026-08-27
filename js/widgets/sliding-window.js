// Sliding window attention. Reused for Longformer (encoder + global tokens) and
// Mistral (causal decoder + rotating KV buffer) with a mode preset.

function initSlidingWindow(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 20;
  let windowSize = opts.windowSize || 4;
  let mode = opts.mode || "decoder"; // "encoder" (Longformer) | "decoder" (Mistral)
  const globalTokens = [0, 10]; // used only in encoder mode

  function visible(center, j) {
    if (mode === "encoder" && globalTokens.includes(j)) return true;
    if (mode === "encoder" && globalTokens.includes(center)) return true;
    return Math.abs(center - j) <= windowSize && (mode === "encoder" || j <= center);
  }

  let center = Math.floor(n * 0.75);

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const pad = 24;
    const cell = (w - pad * 2) / n;
    const y = h / 2;

    for (let j = 0; j < n; j++) {
      const x = pad + j * cell + cell / 2;
      const isCenter = j === center;
      const vis = visible(center, j);
      const evicted = mode === "decoder" && (center - j) > windowSize;
      ctx.beginPath();
      ctx.arc(x, y, isCenter ? 9 : 6, 0, Math.PI * 2);
      if (isCenter) {
        ctx.fillStyle = cssVar("--accent-2");
      } else if (mode === "encoder" && globalTokens.includes(j)) {
        ctx.fillStyle = cssVar("--good");
      } else if (vis) {
        ctx.fillStyle = cssVar("--accent");
      } else if (evicted) {
        ctx.fillStyle = "transparent";
        ctx.strokeStyle = cssVar("--bad");
        ctx.globalAlpha = 0.6;
        ctx.stroke();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = cssVar("--border-strong");
      }
      ctx.fill();
    }

    // window bracket for decoder mode
    if (mode === "decoder") {
      const from = Math.max(0, center - windowSize);
      const x1 = pad + from * cell;
      const x2 = pad + center * cell + cell;
      ctx.strokeStyle = cssVar("--accent");
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = cssVar("--accent");
      ctx.fillRect(x1, y - 22, x2 - x1, 44);
      ctx.globalAlpha = 1;
    }

    ctx.font = "11px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    if (mode === "encoder") {
      ctx.fillText("green = global tokens (always visible)   blue = within local window of the selected token", pad, h - 14);
    } else {
      const evictedCount = Math.max(0, center - windowSize);
      ctx.fillText(`window = ${windowSize} tokens back   —   ${evictedCount} earlier token(s) already evicted from the KV cache`, pad, h - 14);
    }
  }

  bindSlider(root, ".sw-window", ".sw-window-out", (v) => {
    windowSize = Math.round(v);
    draw();
  }, { fmt: (v) => Math.round(v) });

  const posInput = root.querySelector(".sw-pos");
  if (posInput) {
    bindSlider(root, ".sw-pos", ".sw-pos-out", (v) => {
      center = Math.round(v);
      draw();
    }, { fmt: (v) => Math.round(v) });
  }

  window.addEventListener("resize", draw);
  draw();
}

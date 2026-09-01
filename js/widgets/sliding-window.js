// Sliding window attention. Reused for Longformer (encoder + global tokens) and
// Mistral (causal decoder + rotating, size-capped KV cache) via a mode preset.

function initSlidingWindow(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 20;
  let windowSize = opts.windowSize || 4;
  const mode = opts.mode || "decoder"; // "encoder" (Longformer) | "decoder" (Mistral)
  const globalTokens = [0, 10];        // encoder mode only
  let center = mode === "decoder" ? Math.floor(n * 0.75) : Math.floor(n * 0.75);

  function visible(j) {
    if (mode === "encoder") {
      if (globalTokens.includes(j) || globalTokens.includes(center)) return true;
      return Math.abs(center - j) <= windowSize;
    }
    return j <= center && center - j <= windowSize;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const x0 = 30, x1 = w - 24;
    const rowY = 58;
    const at = tokenAxis(ctx, { x0, x1, y: 108, n, step: 2,
      label: mode === "encoder" ? "token position in the document" : "token position in the generated sequence" });

    // highlight band for the window
    const from = mode === "decoder" ? Math.max(0, center - windowSize) : Math.max(0, center - windowSize);
    const to = mode === "decoder" ? center : Math.min(n - 1, center + windowSize);
    const cell = (x1 - x0) / n;
    ctx.save();
    ctx.fillStyle = cssVar("--accent");
    ctx.globalAlpha = 0.13;
    ctx.fillRect(at(from) - cell / 2, rowY - 22, at(to) - at(from) + cell, 44);
    ctx.restore();

    for (let j = 0; j < n; j++) {
      const x = at(j);
      const isGlobal = mode === "encoder" && globalTokens.includes(j);
      const future = mode === "decoder" && j > center;
      const evicted = mode === "decoder" && center - j > windowSize;
      ctx.beginPath();
      ctx.arc(x, rowY, j === center ? 9 : 6, 0, Math.PI * 2);
      if (j === center) {
        ctx.fillStyle = cssVar("--accent-2"); ctx.fill();
      } else if (isGlobal) {
        ctx.fillStyle = cssVar("--good"); ctx.fill();
      } else if (visible(j)) {
        ctx.fillStyle = cssVar("--accent"); ctx.fill();
      } else if (future) {
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = cssVar("--border-strong");
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (evicted) {
        ctx.strokeStyle = cssVar("--bad");
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = cssVar("--border-strong"); ctx.fill();
      }
    }

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`selected token: ${center}`, x0, 26);

    if (mode === "encoder") {
      legend(ctx, x0, h - 42, [
        { color: cssVar("--accent-2"), label: "selected token", shape: "dot" },
        { color: cssVar("--good"), label: "global ([CLS]) — always visible", shape: "dot" },
        { color: cssVar("--accent"), label: "inside the window", shape: "dot" },
        { color: cssVar("--border-strong"), label: "outside — never compared here", shape: "dot" },
      ], 10.5, w - 20);
      caption(ctx, x0, h - 12,
        `window = ±${windowSize} tokens → each token compares against at most ${Math.min(n, 2 * windowSize + 1)} others instead of all ${n}. Cost is linear in length, not quadratic.`);
    } else {
      const cached = Math.min(center + 1, windowSize + 1);
      const evictedCount = Math.max(0, center - windowSize);
      legend(ctx, x0, h - 60, [
        { color: cssVar("--accent-2"), label: "generating now", shape: "dot" },
        { color: cssVar("--accent"), label: "in the KV cache", shape: "dot" },
        { color: cssVar("--bad"), label: "evicted — gone for good", shape: "ring" },
        { color: cssVar("--border-strong"), label: "not generated yet", shape: "ring" },
      ], 10.5, w - 20);
      ctx.textAlign = "left";
      ctx.font = font(12, 700);
      ctx.fillStyle = cssVar("--good");
      ctx.fillText(`KV cache holds ${cached} entries — capped at window+1 = ${windowSize + 1}`, x0, h - 32);
      caption(ctx, x0, h - 12,
        `${evictedCount} earlier token(s) already evicted. Generate 1M more tokens and the cache still holds ${windowSize + 1} — that's the whole point.`);
    }
  }

  bindSlider(root, ".sw-window", ".sw-window-out", (v) => {
    windowSize = Math.round(v);
    draw();
  }, { fmt: (v) => (mode === "encoder" ? "±" : "") + Math.round(v) });

  if (root.querySelector(".sw-pos")) {
    bindSlider(root, ".sw-pos", ".sw-pos-out", (v) => {
      center = Math.round(v);
      draw();
    }, { fmt: (v) => Math.round(v) });
  }

  window.addEventListener("resize", draw);
  draw();
}

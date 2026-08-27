// Position Interpolation vs NTK-aware scaling vs YaRN — one widget, three modes.
// The "quality" bar is an illustrative heuristic (clearly labeled as such), not a
// measured benchmark number: it exists to show the shape of the story (uniform
// compression -> non-uniform -> non-uniform + temperature fix), not exact figures.

function initRopeScaling(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let mode = opts.lockMode || "pi"; // pi | ntk | yarn
  let extension = 4; // 1x .. 8x

  const dims = 24; // illustrative rotary dimension pairs, low freq -> high freq left to right

  function scaledFreq(d, ext) {
    // d in [0,1]: 0 = lowest frequency (long-range), 1 = highest frequency (fine-grained)
    if (mode === "pi") {
      return 1 / ext; // every dimension compressed uniformly
    }
    if (mode === "ntk") {
      // preserve high frequencies, compress low frequencies more
      const t = d; // 0..1
      return 1 / (1 + (ext - 1) * (1 - t));
    }
    // yarn: ntk-by-parts, plus we just show a smoother, more evenly preserved curve
    const t = d;
    const ntkByParts = 1 / (1 + (ext - 1) * Math.pow(1 - t, 1.6));
    return ntkByParts;
  }

  function qualityScore(ext) {
    // illustrative only: PI degrades fastest with extension, NTK slower, YaRN slowest
    const penalty = { pi: 1.0, ntk: 0.55, yarn: 0.3 }[mode];
    return Math.max(0, 1 - penalty * Math.log2(ext) * 0.18);
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const padL = 44, padR = 20, padT = 24, padB = 34;
    const plotW = w - padL - padR;
    const plotH = (h - padT - padB) * 0.62;

    // frequency-preservation curve across dimensions
    ctx.strokeStyle = cssVar("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= dims; i++) {
      const d = i / dims;
      const f = scaledFreq(d, extension); // 0..1, 1 = fully preserved
      const x = padL + d * plotW;
      const y = padT + plotH - f * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.font = "11px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    ctx.fillText("low-frequency dims (long range)", padL, padT + plotH + 16);
    ctx.textAlign = "right";
    ctx.fillText("high-frequency dims (local detail)", padL + plotW, padT + plotH + 16);
    ctx.textAlign = "left";
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText("resolution preserved →", padL, padT - 6);

    // illustrative quality bar
    const q = qualityScore(extension);
    const barY = padT + plotH + 40;
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText(`illustrative quality at ${extension}× extension (not a measured benchmark):`, padL, barY - 6);
    ctx.fillStyle = cssVar("--bg-elevated");
    ctx.fillRect(padL, barY, plotW, 12);
    ctx.fillStyle = q > 0.7 ? cssVar("--good") : q > 0.4 ? cssVar("--accent-2") : cssVar("--bad");
    ctx.fillRect(padL, barY, plotW * q, 12);
  }

  if (opts.lockMode) {
    root.querySelectorAll(".rs-mode").forEach((btn) => btn.remove());
  } else {
    root.querySelectorAll(".rs-mode").forEach((btn) => {
      btn.addEventListener("click", () => {
        mode = btn.dataset.mode;
        root.querySelectorAll(".rs-mode").forEach((b) => b.classList.toggle("on", b === btn));
        draw();
      });
    });
    root.querySelector(`.rs-mode[data-mode="${mode}"]`)?.classList.add("on");
  }

  bindSlider(root, ".rs-ext", ".rs-ext-out", (v) => { extension = v; draw(); }, { fmt: (v) => v + "×" });

  window.addEventListener("resize", draw);
  draw();
}

// KV-cache calculator.
//   bytes = 2 (K and V) × layers × kv_heads × head_dim × context × batch × bytes_per_number
// Default preset (48 layers, 8 KV heads, head_dim 128, bf16, 32,768 tokens)
// reproduces the source lecture's own worked example: ~6.44 GB/user.
// The plot turns the formula into the shape that actually matters: cache grows
// as a straight line in context length, and one 80 GB GPU is a hard ceiling.

function initKVCacheCalc(root) {
  const readout = root.querySelector(".kv-readout");
  const canvas = root.querySelector("canvas");
  const cvs = canvas ? setupCanvas(canvas) : null;
  const GPU_GB = 80; // one H100-class accelerator, for scale
  let layers = 48, kvHeads = 8, headDim = 128, context = 32768, batch = 1, bytesPerNum = 2;

  const bytesAt = (ctxLen) => 2 * layers * kvHeads * headDim * ctxLen * bytesPerNum;
  const fmtGB = (bytes) => (bytes / 1e9).toFixed(2);

  function spanHi(text, cls) {
    const el = document.createElement("span");
    el.className = cls || "hi";
    el.textContent = text;
    return el;
  }

  function draw() {
    if (!cvs) return;
    const ctx = cvs.ctx;
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const maxCtx = 262144;
    const curGB = (bytesAt(context) * batch) / 1e9;
    // scale the axis to the interesting region (the GPU ceiling and the current
    // setting), not to the far corner — otherwise everything readable is squashed
    // into the bottom fifth of the plot. The line simply runs off the top instead.
    const maxGB = Math.max(GPU_GB * 1.5, curGB * 1.35);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 78, r: 26, t: 30, b: 60 },
      x: { min: 0, max: maxCtx, ticks: [0, 65536, 131072, 196608, 262144],
           fmt: (v) => (v === 0 ? "0" : Math.round(v / 1024) + "K"),
           label: "context length (tokens)" },
      y: { min: 0, max: maxGB, ticks: 5, fmt: (v) => v.toFixed(0),
           label: "KV cache across all users (GB)" },
      grid: true,
    });

    // one-GPU ceiling
    if (GPU_GB <= maxGB) {
      ctx.strokeStyle = cssVar("--bad");
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ch.l, ch.py(GPU_GB)); ctx.lineTo(ch.r, ch.py(GPU_GB)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = font(10);
      ctx.textAlign = "right";
      ctx.fillStyle = cssVar("--bad");
      ctx.fillText(`${GPU_GB} GB — one GPU's entire memory`, ch.r - 4, ch.py(GPU_GB) - 6);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(ch.l, ch.t, ch.plotW, ch.plotH);
    ctx.clip();
    ctx.strokeStyle = cssVar("--accent");
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(ch.px(0), ch.py(0));
    ctx.lineTo(ch.px(maxCtx), ch.py((bytesAt(maxCtx) * batch) / 1e9));
    ctx.stroke();
    ctx.restore();
    ctx.strokeStyle = cssVar("--accent-2");
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ch.px(context), ch.b); ctx.lineTo(ch.px(context), ch.py(Math.min(curGB, maxGB))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.beginPath();
    ctx.arc(ch.px(context), ch.py(Math.min(curGB, maxGB)), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = font(11, 700);
    ctx.textAlign = ch.px(context) > ch.l + ch.plotW * 0.7 ? "right" : "left";
    ctx.fillText(`${fmtGB(bytesAt(context) * batch)} GB at ${num(context)} tokens`,
      ch.px(context) + (ctx.textAlign === "right" ? -10 : 10), ch.py(Math.min(curGB, maxGB)) - 12);

    caption(ctx, ch.l, h - 8, `2 × ${layers} layers × ${kvHeads} KV heads × ${headDim} head_dim × context × ${batch} user${batch === 1 ? "" : "s"} × ${bytesPerNum} bytes`);
  }

  function recompute() {
    const perUser = bytesAt(context);
    const total = perUser * batch;
    readout.textContent = "";
    readout.append(
      "per-user cache: ", spanHi(fmtGB(perUser) + " GB"),
      `  ·  total at ${batch} concurrent user${batch === 1 ? "" : "s"}: `,
      spanHi(fmtGB(total) + " GB", total / 1e9 > GPU_GB ? "warn" : "hi"),
      total / 1e9 > GPU_GB ? `  ·  needs more than one ${GPU_GB} GB GPU, before any model weights` : ""
    );
    draw();
  }

  bindSlider(root, ".kv-layers", ".kv-layers-out", (v) => { layers = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-heads", ".kv-heads-out", (v) => { kvHeads = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-dim", ".kv-dim-out", (v) => { headDim = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-ctx", ".kv-ctx-out", (v) => { context = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v).toLocaleString() });
  bindSlider(root, ".kv-batch", ".kv-batch-out", (v) => { batch = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });

  root.querySelectorAll(".kv-precision").forEach((btn) => {
    btn.addEventListener("click", () => {
      bytesPerNum = parseFloat(btn.dataset.bytes);
      root.querySelectorAll(".kv-precision").forEach((b) => b.classList.toggle("on", b === btn));
      recompute();
    });
  });
  root.querySelector('.kv-precision[data-bytes="2"]')?.classList.add("on");

  window.addEventListener("resize", draw);
  recompute();
}

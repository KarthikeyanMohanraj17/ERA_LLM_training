// Scaled dot-product attention, one query at a time, with every intermediate
// number on screen: raw q·k score -> divided by sqrt(d_k) -> softmax weight
// (the bars) -> weighted sum of the values (the output).
// A 6-token toy sequence, causal mask on/off.

function initQKV(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const tokens = ["The", "cat", "sat", "on", "the", "mat"];
  const values = [10, 20, 30, 40, 50, 60]; // toy 1-D "value" per token, so the output is one number
  const dK = 4;                             // toy head dim -> scale factor 1/sqrt(4) = 0.5
  // Fixed toy q·k scores, hand-picked so "sat" attends strongly to "cat".
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

  function compute() {
    const raw = rawScores[query];
    const visible = tokens.map((_, j) => !masked || j <= query);
    const scaled = raw.map((v) => v / Math.sqrt(dK));
    const live = scaled.filter((_, j) => visible[j]);
    const sm = softmax(live);
    const weights = [];
    let k = 0;
    visible.forEach((vis) => weights.push(vis ? sm[k++] : 0));
    const out = weights.reduce((acc, wgt, j) => acc + wgt * values[j], 0);
    return { raw, scaled, visible, weights, out };
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const { raw, scaled, visible, weights, out } = compute();
    const n = tokens.length;

    const ch = chart(ctx, {
      w, h,
      pad: { l: 124, r: 22, t: 78, b: 104 },
      x: { min: -0.5, max: n - 0.5, ticks: [0, 1, 2, 3, 4, 5], fmt: (v) => tokens[Math.round(v)],
           label: "key token j  —  every token the query is compared against" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2),
           label: "softmax weight" },
      grid: true,
    });

    // ---- computation strip above the plot: raw score, then scaled score ----
    ctx.textAlign = "right";
    ctx.font = font(10.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("q·k (raw score)", ch.l - 10, 22);
    ctx.fillText(`÷ √d_k = ÷${Math.sqrt(dK)}`, ch.l - 10, 42);
    ctx.textAlign = "center";
    tokens.forEach((_, j) => {
      const x = ch.px(j);
      if (!visible[j]) {
        ctx.fillStyle = cssVar("--text-faint");
        ctx.font = font(10.5);
        ctx.fillText("masked", x, 22);
        ctx.fillText("—", x, 42);
        return;
      }
      ctx.font = font(11.5);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(raw[j].toFixed(1), x, 22);
      ctx.fillStyle = cssVar("--accent");
      ctx.fillText(scaled[j].toFixed(2), x, 42);
    });

    // ---- bars: the softmax weights ----
    const barW = Math.min(48, (ch.plotW / n) * 0.62);
    tokens.forEach((_, j) => {
      const x = ch.px(j);
      if (!visible[j]) {
        // masked: a hatched stub sitting on the axis, so "blocked" reads as a
        // deliberate zero rather than a missing bar.
        ctx.save();
        ctx.strokeStyle = cssVar("--border-strong");
        ctx.lineWidth = 1;
        for (let o = -barW / 2; o < barW / 2; o += 5) {
          ctx.beginPath();
          ctx.moveTo(x + o, ch.b);
          ctx.lineTo(x + o + 6, ch.b - 8);
          ctx.stroke();
        }
        ctx.restore();
        return;
      }
      const y = ch.py(weights[j]);
      ctx.fillStyle = j === query ? cssVar("--accent-2") : cssVar("--accent");
      ctx.globalAlpha = 0.85;
      roundRect(ctx, x - barW / 2, y, barW, ch.b - y, 4);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.font = font(11.5, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.textAlign = "center";
      ctx.fillText(weights[j].toFixed(2), x, y - 6);
    });

    // ---- the weighted sum, spelled out ----
    const terms = tokens
      .map((_, j) => (visible[j] ? `${weights[j].toFixed(2)}×${values[j]}` : null))
      .filter(Boolean);
    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText(`values v = [${values.join(", ")}]   ·   weights always sum to ${weights.reduce((a, b) => a + b, 0).toFixed(2)}`, ch.l, h - 34);
    ctx.font = font(12.5);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`output = Σ wⱼ·vⱼ = ${terms.join(" + ")} = ${out.toFixed(2)}`, ch.l, h - 14);

    // ---- header: which query, mask state ----
    ctx.textAlign = "left";
    ctx.font = font(12, 700);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`query i = "${tokens[query]}" (token ${query})`, ch.l - 16, 64);
    ctx.font = font(11);
    ctx.fillStyle = masked ? cssVar("--good") : cssVar("--bad");
    ctx.fillText(masked ? "causal mask ON — future tokens excluded before softmax" : "causal mask OFF — the query is reading tokens that haven't happened yet", ch.l + 210, 64);
  }

  bindSlider(root, ".qkv-query", ".qkv-query-out", (v) => {
    query = Math.round(v);
    draw();
  }, { fmt: (v) => tokens[Math.round(v)] });

  bindToggle(root, ".qkv-mask", "mask: ON", "mask: OFF", (v) => {
    masked = v;
    draw();
  }, true);

  window.addEventListener("resize", draw);
  draw();
}

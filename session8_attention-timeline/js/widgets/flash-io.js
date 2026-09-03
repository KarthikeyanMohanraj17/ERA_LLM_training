// FlashAttention: the cost that was never arithmetic.
//
// A naive kernel materialises the T x T score matrix in GPU main memory (HBM)
// and crosses it four times — write scores, read to softmax, write weights,
// read to apply to V. FlashAttention tiles Q/K/V into SRAM and keeps a running
// softmax, so that matrix is never created. Same output, bit for bit.
//
// Every number here is computed from the two sliders. Nothing is asserted.

function initFlashIO(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const BYTES = 2; // fp16
  let T = 4096, d = 64;

  const mb = (b) => b / 1e6;
  function model() {
    const inputs = 3 * T * d * BYTES;      // Q, K, V read in
    const output = T * d * BYTES;          // O written out
    const scores = T * T * BYTES;          // the T x T matrix
    return { inputs, output, scores, naive: inputs + output + 4 * scores, tiled: inputs + output };
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const m = model();
    const max = mb(m.naive) * 1.06;

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`Q, K, V and O together: ${mb(m.inputs + m.output).toFixed(2)} MB`, 20, 20);
    ctx.fillStyle = cssVar("--bad");
    ctx.font = font(11, 700);
    ctx.fillText(`the T×T score matrix alone: ${mb(m.scores).toFixed(2)} MB  —  ${(m.scores / (m.inputs + m.output)).toFixed(1)}× the size of everything it is computed from`, 20, 40);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 178, r: 92, t: 66, b: 96 },
      x: { min: 0, max, ticks: 5, fmt: (v) => Math.round(v) + " MB",
           label: "bytes moved across HBM, one head, one forward pass" },
      y: { min: -0.5, max: 1.5, ticks: [] },
      grid: false, gridX: true,
    });
    [
      { label: "naive kernel", v: mb(m.naive), color: cssVar("--bad") },
      { label: "FlashAttention", v: mb(m.tiled), color: cssVar("--good") },
    ].forEach((r, i) => {
      const y = ch.py(1 - i) - 11;
      ctx.fillStyle = r.color;
      ctx.fillRect(ch.px(0), y, Math.max(2, ch.px(r.v) - ch.px(0)), 22);
      ctx.textAlign = "right";
      ctx.font = font(11.5);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(r.label, ch.l - 10, y + 16);
      ctx.textAlign = "left";
      ctx.font = font(11.5, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.fillText(r.v.toFixed(2) + " MB", ch.px(r.v) + 8, y + 16);
    });

    ctx.textAlign = "left";
    ctx.font = font(12.5, 700);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`${(m.naive / m.tiled).toFixed(0)}× less memory traffic — for mathematically identical output`, ch.l - 158, h - 34);
    caption(ctx, ch.l - 158, h - 14,
      `T = ${num(T)}, head_dim = ${d}, fp16 — the score matrix grows as T², the inputs only as T.`);
  }

  bindSlider(root, ".fa-t", ".fa-t-out", (v) => { T = Math.round(v); draw(); }, { fmt: (v) => Math.round(v).toLocaleString() });
  bindSlider(root, ".fa-d", ".fa-d-out", (v) => { d = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });
  window.addEventListener("resize", draw);
  draw();
}

// The two-route arithmetic demo from the transcript: direct route (visit every
// key) vs regrouped route (precomputed state S). With softmax off, they match
// exactly. With softmax on, they diverge because softmax ties the scores together.

function initSoftmaxCollapse(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const keys = [0.5, 1.0, 1.5];
  const values = [10, 20, 30];
  let q = 2;
  let softmaxOn = false;

  function compute() {
    const scores = keys.map((k) => q * k);
    if (!softmaxOn) {
      const direct = scores.reduce((sum, s, i) => sum + s * values[i], 0);
      const S = keys.reduce((sum, k, i) => sum + k * values[i], 0);
      const regrouped = q * S;
      return { direct, regrouped, weights: null, S };
    } else {
      const w = softmax(scores);
      const direct = w.reduce((sum, wi, i) => sum + wi * values[i], 0);
      // "regrouped" route is not valid under softmax — shown as the same naive
      // linear-fold formula to make the mismatch visible.
      const S = keys.reduce((sum, k, i) => sum + k * values[i], 0);
      const regrouped = q * S / (keys.length); // deliberately naive, to show it no longer applies
      return { direct, regrouped, weights: w, S };
    }
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const { direct, regrouped, weights } = compute();
    ctx.font = "13px var(--mono), monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`query q = ${q}`, 20, 26);
    keys.forEach((k, i) => {
      ctx.fillText(`key${i + 1}=${k}  value${i + 1}=${values[i]}${weights ? `  weight=${weights[i].toFixed(2)}` : ""}`, 20, 50 + i * 20);
    });

    const barY = 130;
    const barMaxW = w - 200;
    const maxVal = Math.max(Math.abs(direct), Math.abs(regrouped), 1);

    ctx.fillStyle = cssVar("--accent");
    ctx.fillText("direct route", 20, barY - 6);
    ctx.fillRect(140, barY - 14, Math.min(barMaxW, (Math.abs(direct) / maxVal) * barMaxW), 16);
    ctx.fillStyle = cssVar("--text");
    ctx.fillText(direct.toFixed(1), 140 + Math.min(barMaxW, (Math.abs(direct) / maxVal) * barMaxW) + 8, barY - 2);

    const match = Math.abs(direct - regrouped) < 0.01;
    ctx.fillStyle = match ? cssVar("--good") : cssVar("--bad");
    ctx.fillText("regrouped (fixed state)", 20, barY + 34);
    ctx.fillRect(140, barY + 26, Math.min(barMaxW, (Math.abs(regrouped) / maxVal) * barMaxW), 16);
    ctx.fillText(regrouped.toFixed(1), 140 + Math.min(barMaxW, (Math.abs(regrouped) / maxVal) * barMaxW) + 8, barY + 38);

    ctx.font = "12px var(--mono), monospace";
    ctx.fillStyle = match ? cssVar("--good") : cssVar("--bad");
    ctx.fillText(match ? "✓ routes match exactly — the state can be folded" : "✗ routes diverge — softmax ties the scores together, folding breaks", 20, barY + 74);
  }

  bindSlider(root, ".sc-q", ".sc-q-out", (v) => {
    q = v;
    draw();
  }, { fmt: (v) => v.toFixed(1) });

  bindToggle(root, ".sc-softmax", "softmax: ON", "softmax: OFF", (v) => {
    softmaxOn = v;
    draw();
  }, false);

  window.addEventListener("resize", draw);
  draw();
}

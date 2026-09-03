// Linear attention's core trick, made fully visible term by term:
//   direct route:    sum over i of  q * k_i * v_i           (needs q for every term)
//   regrouped route: S = sum over i of  k_i * v_i,  then  q * S   (S needs no q at all)
// With softmax off, both routes give the same number — S can be precomputed once
// and reused for any future query. With softmax on, the per-key weight depends on
// every other key's score too (the softmax denominator), so q can no longer be
// factored out — the two routes diverge, and that's exactly why softmax attention
// can't be turned into a running sum the way linear attention can.

function initSoftmaxCollapse(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const keys = [0.5, 1.0, 1.5];
  const values = [10, 20, 30];
  let q = 2;
  let softmaxOn = false;

  function compute() {
    const scores = keys.map((k) => q * k);
    const S = keys.reduce((sum, k, i) => sum + k * values[i], 0); // no q involved
    if (!softmaxOn) {
      const terms = scores.map((s, i) => s * values[i]);
      const direct = terms.reduce((a, b) => a + b, 0);
      const regrouped = q * S;
      return { terms, direct, regrouped, weights: null, S };
    }
    const w = softmax(scores);
    const terms = w.map((wi, i) => wi * values[i]);
    const direct = terms.reduce((a, b) => a + b, 0);
    // Softmax's weights don't factor as q * (fixed per-key number) — there is no
    // valid "S" here. We still compute the naive q*S formula so the mismatch is
    // visible: this is what would happen if you (wrongly) tried to reuse it.
    const regrouped = q * S / keys.length;
    return { terms, direct, regrouped, weights: w, S };
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const { terms, direct, regrouped, weights, S } = compute();
    ctx.textAlign = "left";

    ctx.font = font(12.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("per-key term (top row: this route's weight/score → contribution)", 20, 18);

    const colX = [20, w * 0.37, w * 0.66];
    const rowY = 42;
    keys.forEach((k, i) => {
      const x = colX[i] ?? 20 + i * 150;
      ctx.font = font(13);
      ctx.fillStyle = cssVar("--accent");
      ctx.fillText(`k${i + 1}=${k}  v${i + 1}=${values[i]}`, x, rowY);
      ctx.fillStyle = cssVar("--text-dim");
      if (!softmaxOn) {
        ctx.fillText(`q·k${i + 1} = ${(q * k).toFixed(2)}`, x, rowY + 18);
        ctx.fillStyle = cssVar("--text");
        ctx.fillText(`term = ${(q * k).toFixed(2)} × ${values[i]} = ${terms[i].toFixed(1)}`, x, rowY + 36);
      } else {
        ctx.fillText(`softmax weight = ${weights[i].toFixed(2)}`, x, rowY + 18);
        ctx.fillStyle = cssVar("--text");
        ctx.fillText(`term = ${weights[i].toFixed(2)} × ${values[i]} = ${terms[i].toFixed(1)}`, x, rowY + 36);
      }
    });

    // direct sum
    const sumY = rowY + 66;
    ctx.font = font(12.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("direct route — every term needs q, recomputed from scratch for this query:", 20, sumY);
    ctx.font = font(13.5);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`${terms.map((t) => t.toFixed(1)).join(" + ")} = ${direct.toFixed(1)}`, 20, sumY + 20);

    // state S box — drawn identically regardless of q, to make "computed once" visible
    const sY = sumY + 52;
    ctx.font = font(12.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText(softmaxOn ? "state S = Σ kᵢvᵢ — but softmax weights aren't q × (fixed number), so S alone can't reconstruct the answer:" : "regrouped route — fold the key/value pairs into one state S first, with no q in sight:", 20, sY);
    ctx.font = font(13.5);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`S = (0.5×10)+(1×20)+(1.5×30) = ${S.toFixed(1)}`, 20, sY + 20);
    ctx.fillStyle = cssVar("--text");
    ctx.fillText(`regrouped = q × S = ${q} × ${S.toFixed(1)}${softmaxOn ? " / 3 (naive — doesn't apply under softmax)" : ""} = ${regrouped.toFixed(1)}`, 20, sY + 40);

    // comparison bars — with a scale, so "these two totals differ" is a
    // measurable statement rather than two coloured rectangles
    const barY = sY + 72;
    const barX = 250;
    const barMaxW = w - barX - 80;
    const maxVal = Math.max(Math.abs(direct), Math.abs(regrouped), 1) * 1.05;
    const scale = (v) => (Math.abs(v) / maxVal) * barMaxW;

    ctx.strokeStyle = cssVar("--border");
    ctx.lineWidth = 1;
    ctx.font = font(9.5);
    ctx.textAlign = "center";
    for (let f = 0; f <= 1.0001; f += 0.25) {
      const x = Math.round(barX + f * barMaxW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, barY - 16);
      ctx.lineTo(x, barY + 42);
      ctx.stroke();
      ctx.fillStyle = cssVar("--text-faint");
      ctx.fillText((maxVal * f).toFixed(0), x, barY + 56);
    }

    const match = Math.abs(direct - regrouped) < 0.01;
    [
      { label: "direct route total", v: direct, color: cssVar("--accent"), dy: -16 },
      { label: "regrouped (q × S) total", v: regrouped, color: match ? cssVar("--good") : cssVar("--bad"), dy: 10 },
    ].forEach((r) => {
      ctx.textAlign = "right";
      ctx.font = font(12);
      ctx.fillStyle = r.color;
      ctx.fillText(r.label, barX - 12, r.dy + barY + 13);
      ctx.fillRect(barX, barY + r.dy, scale(r.v), 16);
      ctx.textAlign = "left";
      ctx.font = font(12, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.fillText(r.v.toFixed(1), barX + scale(r.v) + 8, barY + r.dy + 13);
    });

    ctx.textAlign = "left";
    ctx.font = font(12);
    ctx.fillStyle = match ? cssVar("--good") : cssVar("--bad");
    ctx.fillText(match
      ? "✓ match — S can be computed once and reused for any future query"
      : "✗ diverge — softmax's shared denominator means q can't be factored out of the sum",
      20, barY + 82);
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

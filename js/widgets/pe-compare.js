// Sinusoidal PE (a formula — defined at every position, including ones never
// trained on) vs. absolute learned PE (a lookup table — has rows only up to the
// length it was trained at, and literally nothing past that).

function initPECompare(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let trainedMax = 8;
  let showBeyond = false;

  // One representative frequency, so the curve is readable at this scale.
  const sinCurve = (pos) => Math.sin(pos / 2.2);

  // A trained table row is an arbitrary learned vector — not a formula. These
  // are fixed pseudo-values so the picture is stable across redraws.
  const learnedVal = (p) => Math.sin(p * 2.399 + 1.1) * 0.55 + Math.sin(p * 0.87) * 0.3;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const lastPos = showBeyond ? Math.round(trainedMax * 2.5) : trainedMax;
    const step = lastPos > 16 ? 4 : 2;
    const xTicks = [];
    for (let p = 0; p <= lastPos; p += step) xTicks.push(p);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 62, r: 22, t: 46, b: 74 },
      x: { min: 0, max: lastPos, ticks: xTicks, fmt: (v) => String(Math.round(v)),
           label: "token position in the sequence" },
      y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1], fmt: (v) => v.toFixed(1),
           label: "positional value added to dim 0" },
      grid: true,
    });

    // trained region shading + boundary
    ctx.save();
    ctx.fillStyle = cssVar("--accent-soft");
    ctx.globalAlpha = 0.3;
    ctx.fillRect(ch.l, ch.t, ch.px(trainedMax) - ch.l, ch.plotH);
    ctx.globalAlpha = 1;
    ctx.restore();
    ctx.strokeStyle = cssVar("--border-strong");
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(ch.px(trainedMax), ch.t);
    ctx.lineTo(ch.px(trainedMax), ch.b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = font(10);
    ctx.fillStyle = cssVar("--text-faint");
    // keep the label inside the canvas when the boundary sits at the far right
    const bLabelX = ch.px(trainedMax);
    ctx.textAlign = bLabelX > ch.r - 90 ? "right" : "center";
    ctx.fillText(`max trained position = ${trainedMax}`, Math.min(bLabelX, ch.r), ch.t - 8);

    // sinusoidal: one continuous line, evaluated everywhere
    ctx.strokeStyle = cssVar("--good");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let p = 0; p <= lastPos; p += 0.1) {
      const x = ch.px(p), y = ch.py(sinCurve(p));
      if (p === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // learned table: one discrete dot per row that exists
    ctx.fillStyle = cssVar("--accent-2");
    for (let p = 0; p <= trainedMax; p++) {
      ctx.beginPath();
      ctx.arc(ch.px(p), ch.py(learnedVal(p)), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // past the trained length: the table has no row at all
    if (showBeyond) {
      ctx.strokeStyle = cssVar("--bad");
      ctx.lineWidth = 1.5;
      for (let p = trainedMax + 1; p <= lastPos; p++) {
        const x = ch.px(p), y = ch.py(0);
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4);
        ctx.stroke();
      }
      ctx.font = font(10.5);
      ctx.fillStyle = cssVar("--bad");
      ctx.textAlign = "left";
      ctx.fillText("no row in the table → nothing to look up", ch.px(trainedMax) + 10, ch.t + 16);
    }

    legend(ctx, ch.l, h - 16, [
      { color: cssVar("--good"), label: "sinusoidal — a formula, computable at any position", shape: "line" },
      { color: cssVar("--accent-2"), label: "learned table — one trained row per position", shape: "dot" },
    ], 10.5, w - 20);
  }

  bindSlider(root, ".pe-trained", ".pe-trained-out", (v) => {
    trainedMax = Math.round(v);
    draw();
  }, { fmt: (v) => Math.round(v) + " tok" });

  bindToggle(root, ".pe-beyond", "beyond training length: ON", "beyond training length: OFF", (v) => {
    showBeyond = v;
    draw();
  }, false);

  window.addEventListener("resize", draw);
  draw();
}

// Sinusoidal (a real function, defined everywhere) vs. absolute learned positional
// embeddings (a trained lookup table, undefined past the training length).

function initPECompare(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let trainedMax = 8;
  let showBeyond = false;

  function sinCurve(pos) {
    // one representative frequency for illustration
    return Math.sin(pos / 2.2);
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const totalPositions = showBeyond ? trainedMax * 2.5 : trainedMax + 1;
    const padL = 40, padR = 20, padT = 20, padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const midY = padT + plotH / 2;

    ctx.strokeStyle = cssVar("--border");
    ctx.beginPath();
    ctx.moveTo(padL, midY);
    ctx.lineTo(w - padR, midY);
    ctx.stroke();

    // trained-region shading
    const trainedX = padL + (trainedMax / totalPositions) * plotW;
    ctx.fillStyle = cssVar("--accent-soft");
    ctx.globalAlpha = 0.35;
    ctx.fillRect(padL, padT, trainedX - padL, plotH);
    ctx.globalAlpha = 1;

    // sinusoidal curve — always defined
    ctx.strokeStyle = cssVar("--good");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let p = 0; p <= totalPositions; p += 0.2) {
      const x = padL + (p / totalPositions) * plotW;
      const y = midY - sinCurve(p) * (plotH / 2 - 6);
      if (p === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // learned embedding — only defined up to trainedMax, discrete dots, garbage/flat beyond
    ctx.fillStyle = cssVar("--accent-2");
    for (let p = 0; p <= Math.floor(trainedMax); p++) {
      const x = padL + (p / totalPositions) * plotW;
      const y = midY - sinCurve(p * 0.9 + 0.4) * (plotH / 2 - 6) * 0.7;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (showBeyond) {
      // undefined region: draw faint jittery marks to signal "no trained value here"
      ctx.strokeStyle = cssVar("--bad");
      ctx.globalAlpha = 0.6;
      for (let p = Math.ceil(trainedMax) + 0.5; p <= totalPositions; p += 0.6) {
        const x = padL + (p / totalPositions) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, midY - 6);
        ctx.lineTo(x, midY + 6);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    ctx.font = "11px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    ctx.fillText("trained region", padL + 4, padT + 12);
    if (showBeyond) {
      ctx.fillStyle = cssVar("--bad");
      ctx.fillText("learned table: undefined / untrained here", trainedX + 6, midY - plotH / 2 + 4);
    }
    ctx.fillStyle = cssVar("--good");
    ctx.textAlign = "right";
    ctx.fillText("sinusoidal: still computable", w - padR, padT + 12);
  }

  bindSlider(root, ".pe-trained", ".pe-trained-out", (v) => {
    trainedMax = v;
    draw();
  }, { fmt: (v) => Math.round(v) + " tok" });

  bindToggle(root, ".pe-beyond", "beyond training length: ON", "beyond training length: OFF", (v) => {
    showBeyond = v;
    draw();
  }, false);

  window.addEventListener("resize", draw);
  draw();
}

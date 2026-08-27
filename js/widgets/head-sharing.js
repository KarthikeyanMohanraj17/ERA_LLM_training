// MQA / GQA / MHA: N query heads sharing K key-value heads. Reused for both the
// MQA section (locked to kvHeads=1) and the GQA section (kvHeads slider, 1..8).

function initHeadSharing(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const qHeads = 8;
  let kvHeads = opts.lockKV || 2;
  const locked = !!opts.lockKV;

  const colors = ["#6ea8ff", "#ffb454", "#6fcf97", "#f2789f", "#c792ea", "#7fd1d9", "#f5d76e", "#ff8a65"];

  function groupOf(qIdx) {
    return Math.floor(qIdx / (qHeads / kvHeads));
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const boxW = 34, boxH = 26;
    const topY = 30, botY = h - 50;
    const spacingQ = (w - 60) / (qHeads - 1);
    const spacingKV = kvHeads > 1 ? (w - 60) / (kvHeads - 1) : 0;

    ctx.font = "11px var(--mono), monospace";

    // query heads
    const qPos = [];
    for (let i = 0; i < qHeads; i++) {
      const x = 30 + i * spacingQ;
      qPos.push(x);
      ctx.fillStyle = colors[groupOf(i) % colors.length];
      roundRect(ctx, x - boxW / 2, topY, boxW, boxH, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.textAlign = "center";
      ctx.fillText("Q" + i, x, topY + boxH / 2 + 4);
    }

    // kv heads
    const kvPos = [];
    for (let g = 0; g < kvHeads; g++) {
      const x = kvHeads > 1 ? 30 + g * spacingKV : w / 2;
      kvPos.push(x);
    }

    // connecting lines
    for (let i = 0; i < qHeads; i++) {
      const g = groupOf(i);
      ctx.strokeStyle = colors[g % colors.length];
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(qPos[i], topY + boxH);
      ctx.lineTo(kvPos[g], botY);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    for (let g = 0; g < kvHeads; g++) {
      ctx.fillStyle = colors[g % colors.length];
      roundRect(ctx, kvPos[g] - boxW / 2, botY, boxW, boxH, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.fillText("KV" + g, kvPos[g], botY + boxH / 2 + 4);
    }

    ctx.fillStyle = cssVar("--text-faint");
    ctx.textAlign = "left";
    const label = kvHeads === 1 ? "MQA — all query heads share one KV head" : kvHeads === qHeads ? "MHA — every query head has its own KV head" : `GQA — ${qHeads / kvHeads} query heads per shared KV head`;
    ctx.fillText(label, 20, botY + boxH + 22);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillText(`cache size relative to MHA: ${(kvHeads / qHeads).toFixed(2)}×  (${qHeads}/${kvHeads} = ${(qHeads / kvHeads).toFixed(1)}× smaller)`, 20, botY + boxH + 40);
  }

  if (!locked) {
    bindSlider(root, ".hs-kv", ".hs-kv-out", (v) => {
      kvHeads = Math.round(v);
      draw();
    }, { fmt: (v) => Math.round(v) });
  }

  window.addEventListener("resize", draw);
  draw();
}

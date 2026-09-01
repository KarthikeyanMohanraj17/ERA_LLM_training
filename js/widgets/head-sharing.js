// MQA / GQA / MHA: N query heads sharing K key-value heads. Reused for the MQA
// section (locked to kvHeads = 1) and the GQA section (slider, 1..8).
// The lines show who shares what; the bar below turns that into the number that
// actually matters at serving time — bytes of KV cache per token, per layer.

function initHeadSharing(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const qHeads = 8;
  const HEAD_DIM = 128;   // typical
  const BYTES = 2;        // bf16
  let kvHeads = opts.lockKV || 2;
  const locked = !!opts.lockKV;

  const colors = ["#6ea8ff", "#ffb454", "#6fcf97", "#f2789f", "#c792ea", "#7fd1d9", "#f5d76e", "#ff8a65"];
  const groupOf = (qIdx) => Math.floor(qIdx / (qHeads / kvHeads));
  // K and V, one of each per KV head
  const bytesFor = (kv) => 2 * kv * HEAD_DIM * BYTES;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const boxW = 34, boxH = 26;
    const topY = 30, botY = 128;
    const spanX = w - 70;
    const spacingQ = spanX / (qHeads - 1);

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`${qHeads} query heads — every one keeps its own Q projection, unchanged`, 20, 18);

    const qPos = [];
    for (let i = 0; i < qHeads; i++) qPos.push(35 + i * spacingQ);
    const kvPos = [];
    for (let g = 0; g < kvHeads; g++) {
      kvPos.push(kvHeads > 1 ? 35 + g * (spanX / (kvHeads - 1)) : w / 2);
    }

    // lines first so the boxes sit on top of them
    for (let i = 0; i < qHeads; i++) {
      const g = groupOf(i);
      ctx.strokeStyle = colors[g % colors.length];
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(qPos[i], topY + boxH);
      ctx.lineTo(kvPos[g], botY);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.font = font(11);
    for (let i = 0; i < qHeads; i++) {
      ctx.fillStyle = colors[groupOf(i) % colors.length];
      roundRect(ctx, qPos[i] - boxW / 2, topY, boxW, boxH, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.textAlign = "center";
      ctx.fillText("Q" + i, qPos[i], topY + boxH / 2 + 4);
    }
    for (let g = 0; g < kvHeads; g++) {
      ctx.fillStyle = colors[g % colors.length];
      roundRect(ctx, kvPos[g] - boxW / 2, botY, boxW, boxH, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.fillText("KV" + g, kvPos[g], botY + boxH / 2 + 4);
    }

    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`${kvHeads} key/value head${kvHeads === 1 ? "" : "s"} — only these are written to the KV cache`, 20, botY + boxH + 18);
    const name = kvHeads === 1 ? "MQA (1 shared KV head)"
      : kvHeads === qHeads ? "MHA (one KV head per query head)"
      : `GQA (${qHeads / kvHeads} query heads per KV head)`;
    ctx.font = font(12, 700);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(name, 20, botY + boxH + 38);

    // ---- the number that matters: cache bytes per token, per layer ----
    const maxBytes = bytesFor(qHeads);
    const ch = chart(ctx, {
      w, h,
      pad: { l: 132, r: 60, t: h - 96, b: 44 },
      x: { min: 0, max: maxBytes, ticks: [0, maxBytes / 4, maxBytes / 2, (maxBytes * 3) / 4, maxBytes],
           fmt: (v) => num(v), label: `KV cache bytes per token, per layer  (head_dim ${HEAD_DIM}, bf16)` },
      y: { min: -0.5, max: 1.5, ticks: [] },
      grid: false, gridX: true,
    });

    const rows = [
      { label: "MHA baseline", bytes: maxBytes, color: cssVar("--border-strong") },
      { label: name.split(" ")[0], bytes: bytesFor(kvHeads), color: cssVar("--accent") },
    ];
    rows.forEach((r, idx) => {
      const y = ch.py(1 - idx) - 9;
      ctx.fillStyle = r.color;
      ctx.fillRect(ch.l, y, Math.max(2, ch.px(r.bytes) - ch.l), 18);
      ctx.textAlign = "right";
      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(r.label, ch.l - 10, y + 13);
      ctx.textAlign = "left";
      ctx.font = font(11, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.fillText(num(r.bytes) + " B", ch.px(r.bytes) + 8, y + 13);
    });
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

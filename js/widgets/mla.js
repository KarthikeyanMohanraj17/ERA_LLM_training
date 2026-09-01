// MLA vs GQA/MQA — the mechanism-level distinction, plus the number that makes
// the point. GQA/MQA cache FEWER COPIES of the same full-size K/V shape. MLA
// caches ONE JOINTLY-COMPRESSED latent per token and rebuilds full K/V from it
// at attention time. The bar chart stays on screen in both modes, because the
// interesting fact is that MLA and 2-group GQA cache the same amount and only
// one of them keeps MHA-level quality.

function initMLA(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let mode = "gqa"; // gqa | mla
  const colors = ["#6ea8ff", "#ffb454"];
  const HEADS = 8, HEAD_DIM = 128, LATENT = 512; // DeepSeek-V2's actual latent size

  const CACHE = [
    { key: "mha", label: "MHA (8 KV heads)", n: HEADS * HEAD_DIM * 2, note: "best quality, biggest cache" },
    { key: "gqa", label: "GQA (2 KV groups)", n: 2 * HEAD_DIM * 2, note: "smaller cache, measured quality cost" },
    { key: "mla", label: "MLA (latent 512)", n: LATENT, note: "same size as GQA, MHA-level quality reported" },
  ];

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    ctx.textAlign = "left";

    if (mode === "gqa") {
      const groups = 2, boxW = 34, boxH = 24, topY = 42, botY = 138;
      const span = w - 70;
      const qPos = [], gPos = [];
      for (let i = 0; i < HEADS; i++) qPos.push(35 + i * (span / (HEADS - 1)));
      for (let g = 0; g < groups; g++) gPos.push(35 + g * (span / (groups - 1)));

      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText("GQA / MQA: store fewer copies of the same full-size K/V shape", 20, 22);

      for (let i = 0; i < HEADS; i++) {
        const g = i < HEADS / 2 ? 0 : 1;
        ctx.strokeStyle = colors[g];
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(qPos[i], topY + boxH);
        ctx.lineTo(gPos[g], botY);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.font = font(11);
      for (let i = 0; i < HEADS; i++) {
        const g = i < HEADS / 2 ? 0 : 1;
        ctx.fillStyle = colors[g];
        roundRect(ctx, qPos[i] - boxW / 2, topY, boxW, boxH, 5);
        ctx.fill();
        ctx.fillStyle = "#0b0d12";
        ctx.textAlign = "center";
        ctx.fillText("Q" + i, qPos[i], topY + boxH / 2 + 4);
      }
      for (let g = 0; g < groups; g++) {
        ctx.fillStyle = colors[g];
        roundRect(ctx, gPos[g] - boxW / 2, botY, boxW, boxH, 5);
        ctx.fill();
        ctx.fillStyle = "#0b0d12";
        ctx.fillText("KV" + g, gPos[g], botY + boxH / 2 + 4);
      }
      ctx.textAlign = "left";
      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-faint");
      ctx.fillText(`${HEADS / groups} query heads share each full-size KV head — ${groups} real K/V vectors stored per token`, 20, botY + boxH + 20);
    } else {
      const cx = w / 2;
      const fullY = 34, latentY = 96, reconY = 158;
      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.textAlign = "left";
      ctx.fillText("MLA: compress K and V together, cache only the compressed thing, rebuild on demand", 20, 20);

      const box = (y, halfW, fill, label) => {
        ctx.fillStyle = fill;
        roundRect(ctx, cx - halfW, y, halfW * 2, 26, 5);
        ctx.fill();
        ctx.fillStyle = "#0b0d12";
        ctx.font = font(11);
        ctx.textAlign = "center";
        ctx.fillText(label, cx, y + 17);
      };
      const arrow = (y1, y2, label) => {
        ctx.strokeStyle = cssVar("--border-strong");
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, y1); ctx.lineTo(cx, y2 - 5); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - 4, y2 - 9); ctx.lineTo(cx, y2 - 3); ctx.lineTo(cx + 4, y2 - 9); ctx.stroke();
        ctx.textAlign = "left";
        ctx.font = font(10.5);
        ctx.fillStyle = cssVar("--text-faint");
        ctx.fillText(label, cx + 12, (y1 + y2) / 2 + 4);
      };

      box(fullY, 150, cssVar("--accent"), `full K, V  (${HEADS} heads × ${HEAD_DIM} each)`);
      arrow(fullY + 26, latentY, "down-project — compress K and V jointly");
      box(latentY, 78, cssVar("--good"), `latent c  (${LATENT})`);
      ctx.textAlign = "left";
      ctx.font = font(11, 700);
      ctx.fillStyle = cssVar("--good");
      ctx.fillText("← the only thing cached", cx + 90, latentY + 17);
      arrow(latentY + 26, reconY, "up-project — rebuild at attention time");
      ctx.globalAlpha = 0.85;
      box(reconY, 150, cssVar("--accent"), "K, V reconstructed, full size again");
      ctx.globalAlpha = 1;
    }

    // ---- always-visible cache comparison ----
    const maxN = CACHE[0].n;
    const ch = chart(ctx, {
      w, h,
      pad: { l: 152, r: 90, t: h - 116, b: 54 },
      x: { min: 0, max: maxN, ticks: [0, 512, 1024, 1536, 2048], fmt: (v) => num(v),
           label: `numbers cached per token, per layer  (head_dim ${HEAD_DIM}, ${HEADS} heads)` },
      y: { min: -0.5, max: 2.5, ticks: [] },
      grid: false, gridX: true,
    });
    CACHE.forEach((r, idx) => {
      const y = ch.py(2 - idx) - 8;
      const active = r.key === mode || (mode === "gqa" && r.key === "gqa") || (mode === "mla" && r.key === "mla");
      ctx.fillStyle = active ? cssVar("--accent-2") : cssVar("--border-strong");
      ctx.fillRect(ch.l, y, Math.max(2, ch.px(r.n) - ch.l), 16);
      ctx.textAlign = "right";
      ctx.font = font(11, active ? 700 : undefined);
      ctx.fillStyle = active ? cssVar("--text") : cssVar("--text-dim");
      ctx.fillText(r.label, ch.l - 10, y + 12);
      ctx.textAlign = "left";
      ctx.font = font(11, 700);
      ctx.fillStyle = cssVar("--text");
      ctx.fillText(num(r.n), ch.px(r.n) + 8, y + 12);
    });
    caption(ctx, ch.l, h - 8, "GQA-2 and MLA both cache 512 numbers — only MLA reports MHA-level quality at that size.");
  }

  root.querySelectorAll(".mla-mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      mode = btn.dataset.mode;
      root.querySelectorAll(".mla-mode").forEach((b) => b.classList.toggle("on", b === btn));
      draw();
    });
  });
  root.querySelector(`.mla-mode[data-mode="${mode}"]`)?.classList.add("on");

  window.addEventListener("resize", draw);
  draw();
}

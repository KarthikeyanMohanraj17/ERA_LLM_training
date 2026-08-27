// MLA vs GQA/MQA: the mechanism-level distinction. GQA/MQA store fewer *copies*
// of K/V (grouped sharing). MLA stores a single, jointly-compressed low-rank
// latent per token and reconstructs full K/V from it at attention time.

function initMLA(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let mode = "gqa"; // gqa | mla

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    ctx.font = "12px var(--mono), monospace";
    ctx.textAlign = "left";

    if (mode === "gqa") {
      // show 8 heads, 2 shared KV groups, each stored directly
      const heads = 8, groups = 2;
      const boxW = 30, boxH = 22, topY = 30, botY = 100;
      const spacing = (w - 60) / (heads - 1);
      for (let i = 0; i < heads; i++) {
        const x = 30 + i * spacing;
        ctx.fillStyle = cssVar("--accent");
        ctx.globalAlpha = 0.85;
        roundRect(ctx, x - boxW / 2, topY, boxW, boxH, 5);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      const gSpacing = (w - 60) / (groups - 1);
      for (let g = 0; g < groups; g++) {
        const x = 30 + g * gSpacing;
        ctx.fillStyle = cssVar("--accent-2");
        roundRect(ctx, x - boxW / 2, botY, boxW, boxH, 5);
        ctx.fill();
        ctx.fillStyle = "#0b0d12";
        ctx.textAlign = "center";
        ctx.fillText("K/V stored directly", x, botY + boxH + 16);
      }
      ctx.fillStyle = cssVar("--text-dim");
      ctx.textAlign = "left";
      ctx.fillText("GQA/MQA: fewer distinct K/V head groups — each still stored as full-size vectors", 20, botY + boxH + 44);
    } else {
      // show a compression bottleneck: full K/V -> latent -> reconstructed
      const cx = w / 2;
      const fullY = 30, latentY = 90, reconY = 150;
      ctx.fillStyle = cssVar("--accent");
      roundRect(ctx, cx - 100, fullY, 200, 22, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.textAlign = "center";
      ctx.fillText("full K, V per token", cx, fullY + 15);

      ctx.strokeStyle = cssVar("--text-faint");
      ctx.beginPath(); ctx.moveTo(cx, fullY + 22); ctx.lineTo(cx, latentY); ctx.stroke();

      ctx.fillStyle = cssVar("--good");
      roundRect(ctx, cx - 40, latentY, 80, 22, 5);
      ctx.fill();
      ctx.fillStyle = "#0b0d12";
      ctx.fillText("latent c (small)", cx, latentY + 15);
      ctx.fillStyle = cssVar("--text-faint");
      ctx.fillText("← this is what's cached", cx + 140, latentY + 15);

      ctx.strokeStyle = cssVar("--text-faint");
      ctx.beginPath(); ctx.moveTo(cx, latentY + 22); ctx.lineTo(cx, reconY); ctx.stroke();

      ctx.fillStyle = cssVar("--accent");
      ctx.globalAlpha = 0.85;
      roundRect(ctx, cx - 100, reconY, 200, 22, 5);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0b0d12";
      ctx.fillText("K, V reconstructed at read time", cx, reconY + 15);

      ctx.fillStyle = cssVar("--text-dim");
      ctx.textAlign = "left";
      ctx.fillText("MLA: joint low-rank compression, not fewer copies — cache stores only the small latent", 20, reconY + 44);
    }
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

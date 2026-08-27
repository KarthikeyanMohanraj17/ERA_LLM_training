// RoPE rotation: two tokens shown as 2D arrows. Toggle RoPE on/off, drag the
// distance between them, and shift both together to show angle invariance.

function initRope(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let posA = 2;
  let distance = 6;
  let ropeOn = true;
  const theta = 0.35; // radians per position step, for illustration

  function angleFor(pos) {
    return ropeOn ? pos * theta : 0;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const posB = posA + distance;
    const cx1 = w * 0.3, cx2 = w * 0.7, cy = h / 2;
    const r = Math.min(h / 2 - 30, 70);

    function arrow(cx, angle, color, label) {
      const x = cx + r * Math.cos(angle - Math.PI / 2);
      const y = cy + r * Math.sin(angle - Math.PI / 2);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "11px var(--mono), monospace";
      ctx.textAlign = "center";
      ctx.fillText(label, cx, cy + r + 22);
    }

    [cx1, cx2].forEach((cx) => {
      ctx.strokeStyle = cssVar("--border");
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    });

    arrow(cx1, angleFor(posA), cssVar("--accent"), `key @ pos ${posA}`);
    arrow(cx2, angleFor(posB), cssVar("--accent-2"), `query @ pos ${posB}`);

    const angDiff = ropeOn ? (angleFor(posB) - angleFor(posA)) : 0;
    ctx.font = "13px var(--mono), monospace";
    ctx.fillStyle = cssVar("--text-dim");
    ctx.textAlign = "left";
    ctx.fillText(`distance (j - i) = ${distance}`, 20, 24);
    ctx.fillStyle = ropeOn ? cssVar("--good") : cssVar("--bad");
    ctx.fillText(ropeOn ? `relative angle = ${angDiff.toFixed(2)} rad — depends only on distance` : "RoPE off — both arrows point the same way regardless of position", 20, 44);
  }

  bindSlider(root, ".rope-pos", ".rope-pos-out", (v) => { posA = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".rope-dist", ".rope-dist-out", (v) => { distance = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });

  bindToggle(root, ".rope-toggle", "RoPE: ON", "RoPE: OFF", (v) => {
    ropeOn = v;
    draw();
  }, true);

  const shiftBtn = root.querySelector(".rope-shift");
  if (shiftBtn) {
    shiftBtn.addEventListener("click", () => {
      posA += 10;
      draw();
    });
  }

  window.addEventListener("resize", draw);
  draw();
}

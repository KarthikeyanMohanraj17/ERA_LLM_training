// RoPE: rotate the query and the key each by their own position, then take the
// dot product. The absolute rotations cancel and only the difference survives.
//
// Left  — both vectors on ONE circle, with the angle between them shaded. That
//         gap is what the dot product actually sees.
// Right — the resulting score as a function of distance. Shift both positions
//         and the left picture spins while the right plot does not move at all.

function initRope(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let posA = 2;        // key position i
  let distance = 6;    // j - i
  let ropeOn = true;
  const theta = 0.35;  // radians per position step, for illustration

  const angleFor = (pos) => (ropeOn ? pos * theta : 0);
  const scoreAt = (d) => Math.cos(ropeOn ? d * theta : 0); // unit q and k

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const posB = posA + distance;
    const aA = angleFor(posA), aB = angleFor(posB);

    // ---------- left: one circle, two vectors, the angle between them ----------
    const cx = Math.min(150, w * 0.19), cy = h * 0.52, r = Math.min(78, h * 0.3);
    const toXY = (ang) => [cx + r * Math.cos(ang - Math.PI / 2), cy + r * Math.sin(ang - Math.PI / 2)];

    ctx.strokeStyle = cssVar("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    // angle ticks every 45°, so "how far has this rotated" is readable
    for (let a = 0; a < Math.PI * 2; a += Math.PI / 4) {
      const [x1, y1] = toXY(a);
      ctx.beginPath();
      ctx.moveTo(cx + (r - 5) * Math.cos(a - Math.PI / 2), cy + (r - 5) * Math.sin(a - Math.PI / 2));
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // shaded wedge = the relative angle the dot product actually depends on
    if (ropeOn && distance > 0) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r * 0.55, aA - Math.PI / 2, aB - Math.PI / 2, false);
      ctx.closePath();
      ctx.fillStyle = cssVar("--good");
      ctx.globalAlpha = 0.22;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    function arrow(ang, color, label) {
      const [x, y] = toXY(ang);
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
      ctx.font = font(10.5);
      ctx.textAlign = x < cx ? "right" : "left";
      ctx.fillText(label, x + (x < cx ? -8 : 8), y + 4);
    }
    arrow(aA, cssVar("--accent"), `k @ ${posA}`);
    arrow(aB, cssVar("--accent-2"), `q @ ${posB}`);

    ctx.textAlign = "center";
    ctx.font = font(10.5);
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("one 2-D rotary plane", cx, cy + r + 26);

    // ---------- right: score vs distance ----------
    const ch = chart(ctx, {
      w, h,
      pad: { l: Math.max(300, w * 0.42), r: 22, t: 34, b: 72 },
      x: { min: 0, max: 12, ticks: [0, 2, 4, 6, 8, 10, 12], fmt: (v) => String(Math.round(v)),
           label: "distance between the two tokens, j − i" },
      y: { min: -1, max: 1, ticks: [-1, -0.5, 0, 0.5, 1], fmt: (v) => v.toFixed(1),
           label: "attention score q·k" },
      grid: true,
    });
    ctx.strokeStyle = ropeOn ? cssVar("--good") : cssVar("--bad");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let d = 0; d <= 12; d += 0.05) {
      const x = ch.px(d), y = ch.py(scoreAt(d));
      if (d === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // current distance marker
    const mx = ch.px(distance), my = ch.py(scoreAt(distance));
    ctx.strokeStyle = cssVar("--accent-2");
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mx, ch.b);
    ctx.lineTo(mx, my);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.beginPath();
    ctx.arc(mx, my, 5, 0, Math.PI * 2);
    ctx.fill();

    // ---------- readouts ----------
    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`key rotated by ${aA.toFixed(2)} rad · query rotated by ${aB.toFixed(2)} rad`, 20, 20);
    ctx.font = font(12, 700);
    ctx.fillStyle = ropeOn ? cssVar("--good") : cssVar("--bad");
    ctx.fillText(
      ropeOn
        ? `angle between them = (${posB} − ${posA}) × ${theta} = ${((posB - posA) * theta).toFixed(2)} rad   →   q·k = cos(${((posB - posA) * theta).toFixed(2)}) = ${scoreAt(distance).toFixed(3)}`
        : "RoPE off — no rotation, so q·k = 1.00 at every distance: position carries no information at all",
      20, h - 14
    );
  }

  bindSlider(root, ".rope-pos", ".rope-pos-out", (v) => { posA = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".rope-dist", ".rope-dist-out", (v) => { distance = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });
  bindToggle(root, ".rope-toggle", "RoPE: ON", "RoPE: OFF", (v) => { ropeOn = v; draw(); }, true);

  const shiftBtn = root.querySelector(".rope-shift");
  if (shiftBtn) {
    shiftBtn.addEventListener("click", () => {
      const input = root.querySelector(".rope-pos");
      posA = Math.min(parseInt(input.max, 10), posA + 10);
      input.value = posA;
      input.dispatchEvent(new Event("input"));
    });
  }

  window.addEventListener("resize", draw);
  draw();
}

// Delta-rule correction, on one number line so "did the write land on the
// target?" is a thing you can see rather than read.
//   level "basic" -> Schlag et al. 2021: read, compute the delta, write it.
//   level "gated" -> Gated DeltaNet: decay the state first, then correct.
// DeltaNet's own section uses delta-parallel.js — that paper's contribution is
// training speed, not new correction math.

function initDeltaRule(root, opts = {}) {
  const level = opts.level || "basic";
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let current = 40;
  let wanted = 55;
  let gate = 1.0;   // only used when level === "gated"
  let naive = false;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);

    const decayed = level === "gated" ? current * gate : current;
    const delta = wanted - decayed;
    const result = naive ? current + wanted : decayed + delta;

    const lanes = [{ label: "1. read memory", v: current, color: cssVar("--text-dim") }];
    if (level === "gated") lanes.push({ label: `2. decay × ${gate.toFixed(2)}`, v: decayed, color: cssVar("--accent-2") });
    lanes.push({
      label: naive ? "→ after add-only write" : "→ after delta write",
      v: result,
      color: naive ? cssVar("--bad") : cssVar("--good"),
      isResult: true,
    });

    const ch = chart(ctx, {
      w, h,
      pad: { l: 152, r: 30, t: 82, b: 60 },
      x: { min: 0, max: 200, ticks: [0, 50, 100, 150, 200], fmt: (v) => String(Math.round(v)),
           label: "value stored in the memory slot" },
      y: { min: -0.5, max: lanes.length - 0.5, ticks: [] },
      grid: false, gridX: true,
    });

    // the target, as one vertical line across every lane
    ctx.strokeStyle = cssVar("--accent-2");
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ch.px(wanted), ch.t - 8);
    ctx.lineTo(ch.px(wanted), ch.b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = font(10.5);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.textAlign = "center";
    ctx.fillText(`target = ${Math.round(wanted)}`, ch.px(wanted), ch.t - 12);

    lanes.forEach((lane, i) => {
      const y = ch.py(lanes.length - 1 - i);
      ctx.strokeStyle = cssVar("--border");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ch.l, y); ctx.lineTo(ch.r, y); ctx.stroke();

      // the write itself, drawn as the distance actually travelled
      if (lane.isResult) {
        const from = ch.px(naive ? current : decayed);
        ctx.strokeStyle = lane.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(from, y); ctx.lineTo(ch.px(Math.min(200, lane.v)), y); ctx.stroke();
      }
      ctx.fillStyle = lane.color;
      ctx.beginPath();
      ctx.arc(ch.px(Math.min(200, lane.v)), y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.textAlign = "right";
      ctx.font = font(11);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(lane.label, ch.l - 12, y + 4);
      ctx.textAlign = "left";
      ctx.font = font(11, 700);
      ctx.fillStyle = lane.color;
      ctx.fillText(lane.v.toFixed(1), ch.px(Math.min(200, lane.v)) + 10, y + 4);
    });

    // the arithmetic, spelled out
    ctx.textAlign = "left";
    ctx.font = font(12);
    if (naive) {
      ctx.fillStyle = cssVar("--bad");
      ctx.fillText(`add-only write:  ${current} + ${Math.round(wanted)} = ${result.toFixed(1)}   —   overshoots the target by ${(result - wanted).toFixed(1)}`, 20, 26);
    } else {
      ctx.fillStyle = cssVar("--good");
      const pre = level === "gated" ? `decayed = ${current} × ${gate.toFixed(2)} = ${decayed.toFixed(1)}   ·   ` : "";
      ctx.fillText(`${pre}delta = ${Math.round(wanted)} − ${decayed.toFixed(1)} = ${delta.toFixed(1)}`, 20, 26);
      ctx.fillStyle = cssVar("--text-dim");
      ctx.fillText(`write ${decayed.toFixed(1)} + ${delta.toFixed(1)} = ${result.toFixed(1)}  —  lands exactly on the target, by construction`, 20, 46);
    }
  }

  bindSlider(root, ".dr-current", ".dr-current-out", (v) => { current = v; draw(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".dr-wanted", ".dr-wanted-out", (v) => { wanted = v; draw(); }, { fmt: (v) => Math.round(v) });
  if (level === "gated") {
    bindSlider(root, ".dr-gate", ".dr-gate-out", (v) => { gate = v; draw(); }, { fmt: (v) => v.toFixed(2) });
  }
  bindToggle(root, ".dr-naive", "naive add-only", "delta rule", (v) => { naive = v; draw(); }, false);

  window.addEventListener("resize", draw);
  draw();
}

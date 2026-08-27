// Delta-rule correction demo, reused across three sections with growing features:
//   level "basic"  -> delta-rule origin (Schlag et al.): read, delta, correct.
//   level "parallel" -> DeltaNet: same math, note about chunk-parallel training.
//   level "gated"  -> Gated DeltaNet: adds a forget/decay gate before the write.

function initDeltaRule(root, opts = {}) {
  const level = opts.level || "basic";
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  let current = 40;
  let wanted = 55;
  let gate = 1.0; // only used when level === "gated"
  let naive = false;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    ctx.font = "13px var(--mono), monospace";
    ctx.textAlign = "left";

    const decayed = level === "gated" ? current * gate : current;
    const delta = wanted - decayed;
    const result = naive ? current + wanted : decayed + delta;

    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`memory currently returns: ${current}`, 20, 28);
    if (level === "gated") {
      ctx.fillText(`gate = ${gate.toFixed(2)}  ->  decayed value before write: ${decayed.toFixed(1)}`, 20, 50);
    }
    ctx.fillText(`wanted answer: ${wanted}`, 20, level === "gated" ? 72 : 50);

    const y = level === "gated" ? 100 : 78;
    if (naive) {
      ctx.fillStyle = cssVar("--bad");
      ctx.fillText(`naive add-only write: ${current} + ${wanted} = ${result}  (wrong)`, 20, y);
    } else {
      ctx.fillStyle = cssVar("--good");
      ctx.fillText(`delta = wanted − current = ${delta.toFixed(1)}`, 20, y);
      ctx.fillText(`write correction: ${decayed.toFixed(1)} + ${delta.toFixed(1)} = ${result.toFixed(1)}`, 20, y + 22);
    }

    // bar comparison
    const barY = y + 60;
    const maxV = 100;
    const barMaxW = w - 200;
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText("result vs. wanted", 20, barY - 8);
    ctx.fillStyle = naive ? cssVar("--bad") : cssVar("--good");
    ctx.fillRect(150, barY, Math.min(barMaxW, (result / maxV) * barMaxW), 14);
    ctx.fillStyle = cssVar("--accent-2");
    const wantedX = 150 + Math.min(barMaxW, (wanted / maxV) * barMaxW);
    ctx.fillRect(wantedX - 1, barY - 4, 2, 22);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(`result=${result.toFixed(1)}`, 150, barY + 30);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`target=${wanted}`, 150 + 140, barY + 30);
  }

  bindSlider(root, ".dr-current", ".dr-current-out", (v) => { current = v; draw(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".dr-wanted", ".dr-wanted-out", (v) => { wanted = v; draw(); }, { fmt: (v) => Math.round(v) });

  if (level === "gated") {
    bindSlider(root, ".dr-gate", ".dr-gate-out", (v) => { gate = v; draw(); }, { fmt: (v) => v.toFixed(2) });
  }

  bindToggle(root, ".dr-naive", "naive add-only", "delta rule", (v) => {
    naive = v;
    draw();
  }, false);

  window.addEventListener("resize", draw);
  draw();
}

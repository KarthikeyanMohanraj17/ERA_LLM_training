// Attention sinks / StreamingLLM. The mechanism is a softmax accounting problem:
// attention weights must sum to 1, so the model needs somewhere to park the mass
// it does not want to spend. Under a causal mask the first few tokens are the
// natural dumping ground. Evict them and that mass is forced onto real content.

function initAttentionSink(root) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const n = 28;
  const windowSize = 8;
  const sinkCount = 4;
  const SINK_MASS = 0.55; // illustrative: how much attention the sinks absorb
  let position = 20;
  let keepSinks = true;

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);
    const x0 = 30, x1 = w - 24;
    const rowY = 44;
    const at = tokenAxis(ctx, { x0, x1, y: 76, n, step: 4, label: "token position in a long, streaming generation" });

    for (let j = 0; j < n; j++) {
      const x = at(j);
      const isSink = j < sinkCount;
      const inWindow = position - j >= 0 && position - j < windowSize;
      ctx.beginPath();
      ctx.arc(x, rowY, j === position ? 8 : 5, 0, Math.PI * 2);
      if (j === position) {
        ctx.fillStyle = cssVar("--accent-2"); ctx.fill();
      } else if (keepSinks && isSink) {
        ctx.fillStyle = cssVar("--good"); ctx.fill();
      } else if (inWindow) {
        ctx.fillStyle = cssVar("--accent"); ctx.fill();
      } else {
        ctx.strokeStyle = cssVar("--bad");
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.6;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
    legend(ctx, x0, 128, [
      { color: cssVar("--accent-2"), label: "generating now", shape: "dot" },
      { color: cssVar("--good"), label: `${sinkCount} sink tokens, kept forever`, shape: "dot" },
      { color: cssVar("--accent"), label: `sliding window (${windowSize})`, shape: "dot" },
      { color: cssVar("--bad"), label: "evicted", shape: "ring" },
    ], 10.5, w - 20);

    // ---- where the attention mass goes: softmax always sums to 1 ----
    const barL = 30, barR = w - 100, barY = 172, barH = 26;
    const sinkFrac = keepSinks ? SINK_MASS : 0;
    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText("attention weights for the current token — softmax forces these to sum to exactly 1.00", barL, barY - 10);

    ctx.fillStyle = cssVar("--good");
    ctx.fillRect(barL, barY, (barR - barL) * sinkFrac, barH);
    ctx.fillStyle = cssVar("--accent");
    ctx.fillRect(barL + (barR - barL) * sinkFrac, barY, (barR - barL) * (1 - sinkFrac), barH);

    ctx.font = font(11, 700);
    ctx.fillStyle = "#0b0d12";
    ctx.textAlign = "center";
    if (sinkFrac > 0.12) ctx.fillText(`${Math.round(sinkFrac * 100)}% parked on sinks`, barL + (barR - barL) * sinkFrac / 2, barY + 17);
    ctx.fillText(`${Math.round((1 - sinkFrac) * 100)}% on window content`, barL + (barR - barL) * (sinkFrac + (1 - sinkFrac) / 2), barY + 17);

    // 0 / 50 / 100% scale under the bar
    ctx.strokeStyle = cssVar("--border-strong");
    ctx.lineWidth = 1;
    ctx.font = font(10);
    ctx.textAlign = "center";
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const x = Math.round(barL + (barR - barL) * f) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, barY + barH); ctx.lineTo(x, barY + barH + 4); ctx.stroke();
      ctx.fillStyle = cssVar("--text-faint");
      ctx.fillText(Math.round(f * 100) + "%", x, barY + barH + 16);
    });
    ctx.textAlign = "left";
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText("share of this token's total attention weight", barL, barY + barH + 34);

    // ---- verdict ----
    ctx.textAlign = "left";
    ctx.font = font(12, 700);
    if (keepSinks) {
      ctx.fillStyle = cssVar("--good");
      ctx.fillText("sinks kept — the model has its usual dumping ground, generation stays stable", barL, h - 30);
      caption(ctx, barL, h - 12, `cost: ${sinkCount} extra cached tokens, forever. A constant, not a per-token growth.`);
    } else {
      ctx.fillStyle = cssVar("--bad");
      ctx.fillText(`sinks evicted — that ${Math.round(SINK_MASS * 100)}% has nowhere to go and is forced onto window content`, barL, h - 30);
      caption(ctx, barL, h - 12, "Content tokens get weights the model never intended to give them. That is the quality collapse the paper reports.", cssVar("--bad"));
    }
  }

  bindSlider(root, ".as-pos", ".as-pos-out", (v) => { position = Math.round(v); draw(); }, { fmt: (v) => Math.round(v) });
  bindToggle(root, ".as-sinks", "sinks: kept", "sinks: evicted", (v) => { keepSinks = v; draw(); }, true);

  window.addEventListener("resize", draw);
  draw();
}

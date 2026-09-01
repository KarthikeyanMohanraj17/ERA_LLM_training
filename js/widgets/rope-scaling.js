// Position Interpolation vs NTK-aware scaling vs YaRN — one widget, three modes.
//
// Each RoPE dimension pair has a wavelength: how many tokens it takes to
// complete one full rotation. With base 10000, pair 0 turns over in about 6
// tokens (it is what tells "the previous token" from "two tokens ago"), and the
// last pair takes tens of thousands (it only tracks coarse, long-range
// position). Extending context means shrinking rotation angles; the three
// methods differ only in HOW MUCH each dimension gets shrunk.
//
// Everything drawn here is computed from the published rule for each method --
// no invented "quality" score. An earlier version of this widget carried an
// illustrative quality meter and the YaRN section told the reader to draw the
// three-part story's conclusion from it. Numbers this page made up are not
// evidence, so the meter is gone and the resolution curve stands on its own.

function initRopeScaling(root, opts = {}) {
  const canvas = root.querySelector("canvas");
  const { ctx } = setupCanvas(canvas);
  const mode = opts.lockMode || "pi"; // pi | ntk | yarn
  let extension = 4;                  // 1x .. 8x
  const dims = 24;                    // illustrative rotary dimension pairs

  // wavelength of pair i, RoPE base 10000, in tokens
  const wavelength = (i) => 2 * Math.PI * Math.pow(10000, i / (dims - 1));

  const TRAINED_CTX = 4096; // the length the model was trained at, for the by-parts test
  const ALPHA = 1, BETA = 32; // YaRN's own defaults for the LLaMA family

  // fraction of the original rotation speed each dimension keeps
  function keptFrac(d, ext) {
    if (mode === "pi") return 1 / ext;                    // uniform: every dim divided by the same s
    if (mode === "ntk") return 1 / (1 + (ext - 1) * d);   // long-wavelength dims absorb the compression
    // YaRN, "NTK-by-parts": decide per dimension using how many times its
    // wavelength fits inside the trained context. r = ctx / wavelength.
    //   r > BETA  -> the dim already turns over many times inside the trained
    //               window, so it is resolving LOCAL position: leave it alone.
    //   r < ALPHA -> the dim has not completed one turn in the whole window, so
    //               it only encodes coarse range: interpolate it fully.
    //   between   -> linear ramp.
    const r = TRAINED_CTX / wavelength(d * (dims - 1));
    const gamma = Math.min(1, Math.max(0, (r - ALPHA) / (BETA - ALPHA)));
    return gamma * 1 + (1 - gamma) * (1 / ext);
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h, 10);
    clear(ctx, canvas);

    const ch = chart(ctx, {
      w, h,
      pad: { l: 68, r: 26, t: 34, b: 92 },
      x: { min: 0, max: dims - 1, ticks: [0, 6, 12, 18, 23], fmt: (v) => "dim " + Math.round(v),
           label: "rotary dimension pair  —  short wavelength (local detail) → long wavelength (coarse range)" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => Math.round(v * 100) + "%",
           label: "rotation resolution kept" },
      grid: true,
    });

    // Position Interpolation's flat line, drawn as a reference in the other two
    // modes so the comparison doesn't require scrolling between sections.
    if (mode !== "pi") {
      const yPI = ch.py(1 / extension);
      ctx.strokeStyle = cssVar("--text-faint");
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ch.l, yPI); ctx.lineTo(ch.r, yPI); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = cssVar("--accent");
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < dims; i++) {
      const x = ch.px(i), y = ch.py(keptFrac(i / (dims - 1), extension));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const first = keptFrac(0, extension), last = keptFrac(1, extension);
    [[0, first], [dims - 1, last]].forEach(([i, f]) => {
      ctx.fillStyle = i === 0 ? cssVar("--good") : cssVar("--accent-2");
      ctx.beginPath();
      ctx.arc(ch.px(i), ch.py(f), 5, 0, Math.PI * 2);
      ctx.fill();
    });

    legend(ctx, ch.l, ch.b + 44, [
      { color: cssVar("--accent"), label: { pi: "Position Interpolation", ntk: "NTK-aware", yarn: "YaRN" }[mode], shape: "line" },
    ].concat(mode !== "pi" ? [{ color: cssVar("--text-faint"), label: `what Position Interpolation would do (flat ${Math.round((1 / extension) * 100)}%)`, shape: "line" }] : []), 10.5, w - 20);

    ctx.textAlign = "left";
    ctx.font = font(11.5, 700);
    ctx.fillStyle = cssVar("--good");
    ctx.fillText(`dim 0 (rotates every ~${Math.round(wavelength(0))} tokens) keeps ${Math.round(first * 100)}%`, ch.l, ch.b + 66);
    ctx.fillStyle = cssVar("--accent-2");
    ctx.fillText(`dim 23 (rotates every ~${num(wavelength(dims - 1))} tokens) keeps ${Math.round(last * 100)}%`, ch.l + Math.min(360, ch.plotW * 0.5), ch.b + 66);

  }

  root.querySelectorAll(".rs-mode").forEach((btn) => btn.remove());
  bindSlider(root, ".rs-ext", ".rs-ext-out", (v) => { extension = v; draw(); }, { fmt: (v) => v + "×" });

  window.addEventListener("resize", draw);
  draw();
}

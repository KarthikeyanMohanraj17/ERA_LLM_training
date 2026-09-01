// Shared helpers used by every widget module.

function setupCanvas(canvas) {
  // Below about 640px a plot with axis labels has no usable area left — the
  // gutters eat it. Give the canvas a width floor and let it scroll inside its
  // own container instead of squashing every label into the margin.
  if (!canvas.parentElement.classList.contains("canvas-scroll")) {
    const wrap = document.createElement("div");
    wrap.className = "canvas-scroll";
    canvas.parentNode.insertBefore(wrap, canvas);
    wrap.appendChild(canvas);
  }
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d");
  function resize() {
    const w = canvas.clientWidth;
    const h = parseInt(canvas.dataset.h || "220", 10);
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize);
  return { ctx, resize };
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function clear(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.clientWidth, parseInt(canvas.dataset.h || "220", 10));
}

// Wires a <input type=range> to a live numeric readout element and a callback.
function bindSlider(root, selector, outSelector, onChange, opts = {}) {
  const input = root.querySelector(selector);
  const out = outSelector ? root.querySelector(outSelector) : null;
  const fmt = opts.fmt || ((v) => v);
  function update() {
    const v = parseFloat(input.value);
    if (out) out.textContent = fmt(v);
    onChange(v);
  }
  input.addEventListener("input", update);
  update();
  return input;
}

// Wires a toggle button (button.toggle) with on/off text.
function bindToggle(root, selector, onLabel, offLabel, onChange, initial = false) {
  const btn = root.querySelector(selector);
  let state = initial;
  function render() {
    btn.textContent = state ? onLabel : offLabel;
    btn.classList.toggle("on", state);
    onChange(state);
  }
  btn.addEventListener("click", () => {
    state = !state;
    render();
  });
  render();
  return { get: () => state, set: (v) => { state = v; render(); } };
}

function drawGrid(ctx, w, h, step, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= w; x += step) { ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); }
  for (let y = 0; y <= h; y += step) { ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); }
  ctx.stroke();
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/* ==================================================================
   Chart primitives
   ------------------------------------------------------------------
   Canvas 2D parses ctx.font with the CSS font shorthand grammar, and
   that grammar does NOT resolve custom properties. So
       ctx.font = "12px var(--mono), monospace"
   is invalid, the assignment is silently dropped, and text keeps
   rendering at whatever was last valid (default: 10px sans-serif).
   Every size/family choice in every widget was a no-op until this.
   Resolve the family once, then build real font strings from it.
   ================================================================== */

let _monoStack = null;
function monoStack() {
  if (!_monoStack) _monoStack = cssVar("--mono") || "ui-monospace, monospace";
  return _monoStack;
}

// font(12) -> "12px <resolved mono stack>";  font(12, 700) -> bold
function font(size, weight) {
  return `${weight ? weight + " " : ""}${size}px ${monoStack()}`;
}

// Draws a labelled plot box — axis lines, ticks, tick labels, axis titles,
// optional gridlines — and returns data->pixel mappers so callers only ever
// think in data units.
//
//   const ch = chart(ctx, {
//     w, h,
//     x: { min: 0, max: 15, label: "key position j", ticks: [0,5,10,15] },
//     y: { min: 0, max: 1,  label: "attention weight", ticks: 5,
//          fmt: (v) => v.toFixed(2) },
//     grid: true,
//   });
//   ctx.lineTo(ch.px(3), ch.py(0.42));
function chart(ctx, cfg) {
  const pad = Object.assign({ l: 62, r: 22, t: 26, b: 52 }, cfg.pad || {});
  const w = cfg.w, h = cfg.h;
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const X = cfg.x || { min: 0, max: 1 };
  const Y = cfg.y || { min: 0, max: 1 };
  const xSpan = (X.max - X.min) || 1;
  const ySpan = (Y.max - Y.min) || 1;

  const px = (v) => pad.l + ((v - X.min) / xSpan) * plotW;
  const py = (v) => pad.t + plotH - ((v - Y.min) / ySpan) * plotH;

  const faint = cssVar("--text-faint");
  const dim = cssVar("--text-dim");
  const border = cssVar("--border");
  const borderStrong = cssVar("--border-strong");

  function ticksOf(A) {
    if (Array.isArray(A.ticks)) return A.ticks;
    const n = A.ticks || 5;
    const out = [];
    for (let i = 0; i < n; i++) out.push(A.min + (i / (n - 1)) * (A.max - A.min));
    return out;
  }

  ctx.save();
  ctx.lineWidth = 1;

  // gridlines + y ticks
  const yTicks = ticksOf(Y);
  ctx.font = font(10);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  yTicks.forEach((v) => {
    const y = Math.round(py(v)) + 0.5;
    if (cfg.grid !== false) {
      ctx.strokeStyle = border;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
    }
    ctx.strokeStyle = borderStrong;
    ctx.beginPath();
    ctx.moveTo(pad.l - 4, y);
    ctx.lineTo(pad.l, y);
    ctx.stroke();
    ctx.fillStyle = faint;
    ctx.fillText((Y.fmt ? Y.fmt(v) : String(Math.round(v * 100) / 100)), pad.l - 8, y);
  });

  // x ticks
  const xTicks = ticksOf(X);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  xTicks.forEach((v) => {
    const x = Math.round(px(v)) + 0.5;
    if (cfg.gridX) {
      ctx.strokeStyle = border;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + plotH);
      ctx.stroke();
    }
    ctx.strokeStyle = borderStrong;
    ctx.beginPath();
    ctx.moveTo(x, pad.t + plotH);
    ctx.lineTo(x, pad.t + plotH + 4);
    ctx.stroke();
    ctx.fillStyle = faint;
    ctx.fillText(X.fmt ? X.fmt(v) : String(Math.round(v * 100) / 100), x, pad.t + plotH + 7);
  });

  // axis lines
  ctx.strokeStyle = borderStrong;
  ctx.beginPath();
  ctx.moveTo(pad.l + 0.5, pad.t);
  ctx.lineTo(pad.l + 0.5, pad.t + plotH + 0.5);
  ctx.lineTo(pad.l + plotW, pad.t + plotH + 0.5);
  ctx.stroke();

  // axis titles
  ctx.fillStyle = dim;
  ctx.font = font(11);
  if (X.label) {
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(X.label, pad.l + plotW / 2, pad.t + plotH + 24);
  }
  if (Y.label) {
    ctx.save();
    ctx.translate(Math.max(12, pad.l - 46), pad.t + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(Y.label, 0, 0);
    ctx.restore();
  }
  ctx.restore();
  ctx.textBaseline = "alphabetic";

  return { px, py, plotW, plotH, l: pad.l, t: pad.t, r: pad.l + plotW, b: pad.t + plotH };
}

// Horizontal legend row. items: [{color, label, shape}] where shape is
// "line" | "dot" | "box" | "ring". Returns the x it ended at.
function legend(ctx, x, y, items, size = 10.5, maxX = Infinity) {
  ctx.save();
  ctx.font = font(size);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  let cx = x;
  items.forEach((it) => {
    // wrap to a second row rather than running off the canvas
    const need = 20 + ctx.measureText(it.label).width + 20;
    if (cx > x && cx + need > maxX) { cx = x; y += 17; }
    ctx.strokeStyle = it.color;
    ctx.fillStyle = it.color;
    ctx.lineWidth = 2;
    if (it.shape === "line") {
      ctx.beginPath();
      ctx.moveTo(cx, y);
      ctx.lineTo(cx + 14, y);
      ctx.stroke();
      cx += 19;
    } else if (it.shape === "ring") {
      ctx.beginPath();
      ctx.arc(cx + 6, y, 4.5, 0, Math.PI * 2);
      ctx.stroke();
      cx += 17;
    } else if (it.shape === "box") {
      ctx.fillRect(cx, y - 5, 11, 10);
      cx += 16;
    } else {
      ctx.beginPath();
      ctx.arc(cx + 5, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      cx += 16;
    }
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(it.label, cx, y);
    cx += ctx.measureText(it.label).width + 20;
  });
  ctx.restore();
  ctx.textBaseline = "alphabetic";
  return cx;
}

// One line of small explanatory text under a chart.
function caption(ctx, x, y, text, color) {
  ctx.save();
  ctx.font = font(10.5);
  ctx.textAlign = "left";
  ctx.fillStyle = color || cssVar("--text-faint");
  ctx.fillText(text, x, y);
  ctx.restore();
}

// Thousands separators for the big numbers in the cache widgets.
function num(v) {
  return Math.round(v).toLocaleString();
}

// A one-dimensional position axis for the "row of tokens" widgets (sliding
// window, attention sinks, chunked training). Returns at(j) -> pixel centre of
// token j, so the caller can place dots on it.
function tokenAxis(ctx, o) {
  const cell = (o.x1 - o.x0) / o.n;
  const at = (j) => o.x0 + j * cell + cell / 2;
  ctx.save();
  ctx.strokeStyle = cssVar("--border-strong");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(o.x0, o.y + 0.5);
  ctx.lineTo(o.x1, o.y + 0.5);
  ctx.stroke();
  ctx.font = font(10);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = o.step || Math.max(1, Math.round(o.n / 8));
  for (let j = 0; j < o.n; j += step) {
    const x = Math.round(at(j)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, o.y);
    ctx.lineTo(x, o.y + 4);
    ctx.stroke();
    ctx.fillStyle = cssVar("--text-faint");
    ctx.fillText(String(j), x, o.y + 7);
  }
  if (o.label) {
    ctx.font = font(11);
    ctx.fillStyle = cssVar("--text-dim");
    ctx.fillText(o.label, (o.x0 + o.x1) / 2, o.y + 24);
  }
  ctx.restore();
  ctx.textBaseline = "alphabetic";
  return at;
}

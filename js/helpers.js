// Shared helpers used by every widget module.

function setupCanvas(canvas) {
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

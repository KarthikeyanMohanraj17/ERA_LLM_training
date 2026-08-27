// KV-cache calculator: bytes = 2 * layers * kv_heads * head_dim * T * batch * bytes_per_number.
// Default preset (48 layers, 8 kv heads, head_dim 128, bf16, T=32768, batch=1) reproduces
// the transcript's own worked example: ~6.44 GB/user, ~51.54 GB at 8 concurrent users.

function initKVCacheCalc(root) {
  const readout = root.querySelector(".kv-readout");
  let layers = 48, kvHeads = 8, headDim = 128, context = 32768, batch = 1, bytesPerNum = 2;

  function fmtGB(bytes) {
    return (bytes / 1e9).toFixed(2);
  }

  function recompute() {
    const perUser = 2 * layers * kvHeads * headDim * context * bytesPerNum;
    const total = perUser * batch;

    readout.textContent = "";
    readout.append(
      "per-user cache: ",
      spanHi(fmtGB(perUser) + " GB"),
      `  ·  total at ${batch} concurrent user${batch === 1 ? "" : "s"}: `,
      spanHi(fmtGB(total) + " GB")
    );
  }

  function spanHi(text) {
    const el = document.createElement("span");
    el.className = "hi";
    el.textContent = text;
    return el;
  }

  bindSlider(root, ".kv-layers", ".kv-layers-out", (v) => { layers = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-heads", ".kv-heads-out", (v) => { kvHeads = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-dim", ".kv-dim-out", (v) => { headDim = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });
  bindSlider(root, ".kv-ctx", ".kv-ctx-out", (v) => { context = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v).toLocaleString() });
  bindSlider(root, ".kv-batch", ".kv-batch-out", (v) => { batch = Math.round(v); recompute(); }, { fmt: (v) => Math.round(v) });

  root.querySelectorAll(".kv-precision").forEach((btn) => {
    btn.addEventListener("click", () => {
      bytesPerNum = parseFloat(btn.dataset.bytes);
      root.querySelectorAll(".kv-precision").forEach((b) => b.classList.toggle("on", b === btn));
      recompute();
    });
  });
  root.querySelector('.kv-precision[data-bytes="2"]')?.classList.add("on");

  recompute();
}

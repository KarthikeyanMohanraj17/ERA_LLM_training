document.addEventListener("DOMContentLoaded", () => {
  initNav();

  initQKV(document.getElementById("w-qkv"));
  initPECompare(document.getElementById("w-pe"));
  initSparsePattern(document.getElementById("w-sparse"));
  initHeadSharing(document.getElementById("w-mqa"), { lockKV: 1 });
  initSlidingWindow(document.getElementById("w-longformer"), { mode: "encoder", windowSize: 3 });
  initSoftmaxCollapse(document.getElementById("w-linear"));
  initDeltaRule(document.getElementById("w-delta-origin"), { level: "basic" });
  initRope(document.getElementById("w-rope"));
  initAlibi(document.getElementById("w-alibi"));
  initFlashIO(document.getElementById("w-flash"));
  initHeadSharing(document.getElementById("w-gqa"), {});
  initRopeScaling(document.getElementById("w-pi"), { lockMode: "pi" });
  initRopeScaling(document.getElementById("w-ntk"), { lockMode: "ntk" });
  initRopeScaling(document.getElementById("w-yarn"), { lockMode: "yarn" });
  initYarnTemperature(document.getElementById("w-yarn-temp"));
  initSlidingWindow(document.getElementById("w-mistral"), { mode: "decoder", windowSize: 4 });
  initAttentionSink(document.getElementById("w-sinks"));
  initMLA(document.getElementById("w-mla"));
  initDeltaParallel(document.getElementById("w-deltanet"));
  initDeltaRule(document.getElementById("w-gated-deltanet"), { level: "gated" });
  initTopK(document.getElementById("w-nsa"), { defaultK: 4, branches: true });
  initTopK(document.getElementById("w-dsa"), { defaultK: 3, unit: "token", unitScale: 128 });
  initDrope(document.getElementById("w-drope"));
  initKVCacheCalc(document.getElementById("w-kvcache"));
  initDepthSchedule(document.getElementById("w-lightninglm"));
});

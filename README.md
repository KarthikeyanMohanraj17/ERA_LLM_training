# How Attention Actually Got Here

An interactive, chronological tour of every major attention mechanism, built for the
Session 8 assignment. Static site — no build step, no framework, no backend.

## What this is

Standard scaled dot-product attention (2017) opens the page. Every section after that is
ordered by **real launch date**, not by teaching order or topic grouping, so the story reads
the way the field actually moved: a mechanism solves a real problem, creates a new cost, and
a later mechanism responds to that cost.

## Running it locally

```
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Sources for every date used in the chronology

All dates are the arXiv `v1` submission date or the official release/blog date, not a later
conference/camera-ready date, unless noted otherwise. Every date below was checked directly
against the primary source — not recalled from a language model's memory — before it was
used in the app.

| Order | Mechanism | Source | Date |
|---|---|---|---|
| baseline | Scaled dot-product attention | Vaswani et al., "Attention Is All You Need," [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Jun 2017 |
| 1 | Sinusoidal positional encoding | Same paper as above | Jun 2017 |
| 2 | Absolute learned positional embeddings | Radford et al. (OpenAI), "Improving Language Understanding by Generative Pre-Training" (GPT-1), [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf); adopted by BERT, Devlin et al., [arXiv:1810.04805](https://arxiv.org/abs/1810.04805); footnote — Gehring et al., [arXiv:1705.03122](https://arxiv.org/abs/1705.03122) predates both by roughly a year in a non-attention (conv seq2seq) architecture | Jun 2018 (footnote: May 2017) |
| 3 | Sparse Transformers | Child, Gray, Radford, Sutskever (OpenAI), "Generating Long Sequences with Sparse Transformers," [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) | Apr 2019 |
| 4 | Multi-Query Attention (MQA) | Shazeer (Google), "Fast Transformer Decoding: One Write-Head is All You Need," [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | Nov 2019 |
| 5 | Sliding window attention (origin) | Beltagy, Peters, Cohan (AI2), "Longformer: The Long-Document Transformer," [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) | Apr 2020 |
| 6 | Linear attention | Katharopoulos, Vyas, Pappas, Fleuret, "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) | Jun 2020 |
| 7 | The delta rule (origin) | Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers," [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) | Feb 2021 |
| 8 | RoPE | Su, Lu, Pan, Murtadha, Wen, Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | Apr 2021 |
| 9 | ALiBi | Press, Smith, Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation," [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | Aug 2021 |
| 10 | Grouped-Query Attention (GQA) | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (Google Research), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | May 2023 |
| 11 | Position Interpolation (bonus mechanism, not covered in the original lecture) | Chen, Wong, Chen, Tian (Meta), "Extending Context Window of Large Language Models via Positional Interpolation," [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) | Jun 2023 |
| 12 | NTK-aware RoPE scaling | u/bloc97, r/LocalLLaMA community post (not peer-reviewed); related independent work by kaiokendev the same month | ~Jun 30, 2023 (approximate — see note below) |
| 13 | YaRN | Peng, Quesnelle, Fan, Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | Aug 2023 |
| 14 | Sliding window attention (decoder + KV-cache application) | Mistral AI, [official blog](https://mistral.ai/news/announcing-mistral-7b/); Jiang et al., "Mistral 7B," [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) | Sep 2023 (blog) / Oct 2023 (paper) |
| 15 | Attention sinks / StreamingLLM | Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks," [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | Sep 2023 |
| 16 | Multi-head Latent Attention (MLA) | DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | May 2024 |
| 17 | DeltaNet (parallel-training algorithm for the delta rule) | Yang, Wang, Zhang, Shen, Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | Jun 2024 |
| 18 | Gated DeltaNet | Yang, Kautz, Hatamizadeh (NVIDIA), "Gated Delta Networks: Improving Mamba2 with Delta Rule," [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | Dec 2024 |
| 19 | Native Sparse Attention (NSA) | DeepSeek-AI, Peking University, University of Washington, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention," [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | Feb 2025 |
| 20 | DeepSeek Sparse Attention (DSA) | DeepSeek-AI, [DeepSeek-V3.2-Exp release](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | Sep 2025 |
| 21 | DroPE | Gelberg, Eguchi, Akiba, Cetin (Sakana AI), "DroPE: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings," [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) | Dec 2025 |
| 22 | LightningLM 0.1V (case study) | Shravan (The School of AI, ERA V4 cohort), "Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling," [arXiv:2606.07404](https://arxiv.org/abs/2606.07404) | Jun 2026 |

## Corrections found by checking against primary sources

The brief said to check every date against the actual paper and not to accept anything an
agent asserted confidently. Doing that turned up six errors — five of them in this app's own
earlier draft, one in the source lecture material. Each was verified by fetching the primary
source directly, not by asking a model.

### 1. Numbers attributed to a real paper that are not in it — the worst one

An earlier version of the LightningLM widget drew two bars captioned *"interpolated between
LightningLM 0.1V's two reported points (1/8 G → 8/8 G: 8.0× cache, ~1.41× compute)."*

I fetched the full text of [arXiv:2606.07404](https://arxiv.org/abs/2606.07404) and searched it:

| String | Hits in the paper |
|---|---|
| `1.41` | 0 |
| `8.0×` / `8.0x` | 0 |
| `KV cache` | 0 |

Worse than absent — contradicted. The paper's Table 5 fixes the D:G ratio at 3:1 for every
stage it trains (6 D : 2 G at eight layers, 15 D : 5 G at twenty) and never sweeps it. Neither
endpoint of that widget was ever trained or measured. The word *"reported"* in the caption made
the false attribution explicit, on a real, citable paper by a named real person.

The widget now shows only what is derivable — if only G layers hold a cache, cache share is
`G ÷ 8` — labelled as this page's arithmetic, attributed to nobody.

### 2. The "8K → 256K, 32×" figure — an error in the source lecture

This figure was attributed to LightningLM in the lecture material. It is not in the paper.
Searching the full text returns zero hits for `256K` and zero for `32×`. The paper states the
opposite outright:

> "The production models here were trained and evaluated at 4K and 8K, the longer reaches were
> not tested in this work."

The 8× and 80× extraction factors it *does* cite belong to Gelberg et al. 2025 (the DroPE
authors), which from an 8K base would be roughly 64K and 640K — neither of which is 256K. The
earlier draft of this app flagged the figure as "could not be independently confirmed," which
was too soft: the paper does not fail to confirm it, it disclaims it.

### 3. Learned positional embeddings were not a 2018 fix for a 2017 problem

The app framed absolute learned PE as GPT-1 (Jun 2018) solving a limitation of sinusoidal
encoding. That causal chain did not happen. Both options are in the 2017 Transformer paper,
tested against each other. From [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) §3.5,
verified verbatim:

> "We also experimented with using learned positional embeddings instead, and found that the
> two versions produced nearly identical results (see Table 3 row (E))."

They chose sinusoidal on a hypothesis — *"because it may allow the model to extrapolate to
sequence lengths longer than the ones encountered during training"* — not on a measurement.
The section now says what actually happened: two equal-scoring options, and the field picked
the pragmatic one. The learned-table construction is also older than the app implied — Gehring
et al., [arXiv:1611.02344](https://arxiv.org/abs/1611.02344), Nov 2016, predating the
Transformer entirely.

### 4. DSA selects tokens, not blocks

The app described DeepSeek Sparse Attention as ranking and selecting *blocks*, which erases the
one thing distinguishing it from NSA. V3.2-Exp's lightning indexer scores every preceding
*token* and the selector keeps the top-k individual tokens (k = 2,048). NSA selects blocks;
DSA selects tokens. Corrected in both the prose and the widget.

### 5. DroPE's title, and a missing ancestor

The arXiv title is *"Extending the Context of Pretrained LLMs by Dropping Their Positional
Embeddings"* — there is no `DroPE:` prefix. "DroPE" is a backronym coined inside the abstract.

More substantively, the app presented "is the rotation worth keeping at all?" as a question
first asked in 2025. It was not. Haviv et al. asked it in
[Mar 2022](https://arxiv.org/abs/2203.16634) and Kazemnejad et al. answered it at NeurIPS 2023
([arXiv:2305.19466](https://arxiv.org/abs/2305.19466)), finding that NoPE *beat* ALiBi, RoPE
and T5-relative on length generalization — the exact axis this app plots. DroPE's own paper
cites both. NoPE is now a section of its own, and DroPE is framed as the fix for NoPE's real
weakness rather than as an idea from nowhere.

### 6. Arithmetic errors in worked examples

Recomputed by hand; all four were wrong.

| Where | Was | Correct |
|---|---|---|
| DroPE, `80 mod 2π` | 5.13 | **4.602** |
| DroPE, `400 mod 2π` | 0.35 | **4.159** |
| YaRN, `softmax([4,2,1,.5,0,0,0,0])` top weight | 0.83 | **0.776** |
| YaRN, same after ÷0.5 temperature | 0.68 | **0.660** |
| DSA, ranking share of cost | "6% of the remaining bill" | **33%** of the new bill (6% of the *dense* bill) |

The DroPE example's punchline depended on `400 mod 2π` being 0.35 so that the same angle
recurred at position 35. It does not, so that coincidence never existed; the example now uses
the real recurrence at position ~416.

### Also removed: invented "quality" numbers

The three RoPE-scaling widgets carried an illustrative quality meter, and the YaRN readout told
the reader to *"compare the illustrative quality bars — that side-by-side gap is the whole point
of this three-part story."* That instructed the reader to draw the section's main conclusion
from three numbers this page made up. The meter is gone. The YaRN curve was also wrong — it
used `1/(1 + (s−1)·d^1.6)`, the NTK curve with an invented exponent — and now implements the
published NTK-by-parts rule, keyed on each dimension's wavelength relative to the trained
context.

## Mechanisms added beyond the assignment's minimum list

Each was held to the same standard as the rest of the timeline: v1 date fetched from the arXiv
API, title and author list verified, and a stated motivation / mechanism / advantage / cost /
timeline slot.

| Order | Mechanism | Source | v1 date (verified) | Why it belongs |
|---|---|---|---|---|
| 02 | Relative position representations | Shaw, Uszkoreit, Vaswani — [arXiv:1803.02155](https://arxiv.org/abs/1803.02155); refined by Transformer-XL [arXiv:1901.02860](https://arxiv.org/abs/1901.02860), simplified by T5 [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) | 6 Mar 2018 | RoPE and ALiBi are both relative schemes, and without this the timeline gives them no ancestor. ALiBi in particular *is* T5's scalar logit bias with the learned value replaced by a fixed slope — unreadable as an invention out of nowhere. |
| 11 | FlashAttention | Dao, Fu, Ermon, Rudra, Ré — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | 27 May 2022 | The largest gap in the original chronology: 21 months with nothing in it. It is also the page's one counterexample — exact attention made dramatically cheaper at zero quality cost — which reframes every approximation around it. Much of the "quadratic cost" being approximated away was memory traffic, and that part had an exact answer. |
| 13 | NoPE | Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy — [arXiv:2305.19466](https://arxiv.org/abs/2305.19466); earlier evidence Haviv et al. [arXiv:2203.16634](https://arxiv.org/abs/2203.16634) | 31 May 2023 | The control condition for six sections of position engineering, and DroPE's direct ancestor — cited in DroPE's own paper. |
| 20 | Mamba-2 | Dao, Gu — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060); predecessor Mamba [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | 31 May 2024 | Gated DeltaNet's literal title is *"Improving Mamba2 with Delta Rule"*. Without this section that title cannot be read, and the convergence of the state-space and linear-attention branches is invisible. |

## Notes on the harder-to-source claims

- **NTK-aware scaling's exact date**: the community post that introduced this is on Reddit
  (r/LocalLLaMA, user bloc97). Reddit's own post timestamp wasn't directly fetchable during
  research; the ~June 30, 2023 date is corroborated by secondary sources (a GitHub archive of
  ML paper notes, a Hugging Face Transformers PR referencing it, and an EleutherAI blog post),
  not confirmed first-hand from Reddit's metadata. Treated as approximate throughout the app.
- **The delta rule vs. "DeltaNet"**: these are commonly conflated. The delta rule itself
  originates with Schlag, Irie, and Schmidhuber (Feb 2021). "DeltaNet" — the name most people
  actually mean today — comes from Yang et al.'s June 2024 paper, which is a parallel-training
  algorithm for that same recurrence, not the origin of the idea. Both get their own timeline
  entry above so the correction is visible, not just asserted.
- **Sliding window attention has two distinct contributors**: Longformer (Beltagy et al.,
  2020) originated the mechanism for encoder-style long-document tasks with designated global
  tokens. Mistral 7B (2023) applied the same core idea inside causal, KV-cached decoder-only
  generation with a rotating buffer. Both get separate entries, cross-referenced.
- **MLA vs. GQA/MQA**: confirmed directly against the DeepSeek-V2 paper text. MLA performs
  joint low-rank compression of keys and values into one shared latent, reconstructed at
  attention time — a different mechanism from GQA/MQA's "fewer stored copies," not a rebrand
  of it. The paper itself states GQA/MQA underperform full MHA while MLA does not.
- **DroPE and "LightningLM V4"**: both are real and independently verifiable, but the naming
  in the source lecture material is imprecise, and it matters because LightningLM is this
  course's own project. DroPE originates from Sakana AI (Dec 2025), not from LightningLM's
  authors. "LightningLM" is a 120B-parameter sparse MoE built by The School of AI's ERA V4
  cohort — its actual version string is **0.1V**, not "V4"; "V4" names the cohort, not a model
  version. The DDDGDDDG layer schedule and the "Memory Stream" cross-chunk mechanism are
  confirmed in the paper (§3.2, §5.3); DroPE is confirmed as applied to LightningLM, but as a
  downstream application of Sakana's technique, not something invented in-house.
- **The "8K → 256K, 32× extension" figure** attributed to LightningLM in the source lecture
  could not be independently confirmed during this app's research pass — the relevant section
  of the paper was only partially retrievable. The app presents this figure as reported, not
  as independently verified, and flags exactly where to re-check it (arXiv:2606.07404, §5.3).

## Illustrative vs. measured numbers

A few widgets (RoPE-scaling quality bars, the attention-sink stability curve, the
LightningLM cache/compute multipliers) use simple illustrative formulas to show the *shape*
of a real, reported finding — they are explicitly labeled "illustrative" in the UI and are
not measured benchmark numbers. The KV-cache calculator is the one widget that computes an
exact, checkable formula: `bytes = 2 × layers × kv_heads × head_dim × context × batch ×
bytes_per_number`. Its defaults (48 layers, 8 KV heads, head_dim 128, bf16, 32,768-token
context) reproduce the source lecture's own worked example: ≈6.44 GB for one user, ≈51.54 GB
at 8 concurrent users.

## Every section has a worked example

Each of the 24 sections carries a short hand-computable dry run in the same
**Given → numbered steps → Answer** shape, using small numbers you can check on paper.
They are there so the mechanism is a calculation you can follow, not a claim you have to
accept: one query attending over two keys, the exact bytes MQA saves per token, the
arithmetic that shows linear attention's regrouping trick giving the same answer twice,
the 51.54 GB the KV-cache formula lands on.

## Charts, axes, and one canvas gotcha

Every plot goes through `chart()` in `js/helpers.js`, which draws the axis lines, ticks,
tick labels and rotated axis titles and hands back `px(v)` / `py(v)` mappers, so widgets
work in data units and every chart is labelled the same way. `legend()`, `caption()` and
`tokenAxis()` cover the rest.

**The gotcha, worth knowing before editing any widget:** canvas parses `ctx.font` with the
CSS font shorthand grammar, and that grammar does **not** resolve custom properties. So

```js
ctx.font = "12px var(--mono), monospace";   // invalid -> silently ignored
```

leaves the font at whatever was last valid — in practice `10px sans-serif` for the whole
page. Every size and family choice written that way is a no-op. Use the `font(size, weight)`
helper instead, which resolves `--mono` once and builds a real font string.

Canvases also carry `min-width: 640px` inside an `overflow-x: auto` wrapper, so on narrow
screens they scroll at a readable size rather than squashing axis labels into the gutter.

## Structure

```
index.html            single scrolling page, one section per mechanism, chronological
styles.css            shared visual system (dark/light aware)
data/timeline.json    citation reference data — kept alongside the README's source table;
                      not fetched at runtime (the sections in index.html are the live copy)
js/helpers.js         canvas setup, chart()/legend()/caption()/tokenAxis() primitives,
                      font(), slider and toggle bindings, softmax
js/nav.js             builds the left-hand timeline rail and scroll-spy highlighting
js/main.js            wires every widget to its container on page load
js/widgets/*.js       one module per interactive widget, several reused across sections
                      with different parameters (head-sharing for MQA+GQA, sliding-window
                      for Longformer+Mistral, delta-rule for delta-rule-origin+Gated DeltaNet,
                      rope-scaling for PI+NTK+YaRN, top-k for NSA+DSA — with NSA additionally
                      showing its three parallel branches). alibi.js, drope.js, and
                      delta-parallel.js are each dedicated to one section — ALiBi and DroPE had
                      no widget at all in the first pass, and DeltaNet's parallel-training
                      section originally reused delta-rule.js, which showed the correction math
                      again instead of the actual point of that paper (training in parallel).
```

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

## Structure

```
index.html          single scrolling page, one section per mechanism, chronological
styles.css           shared visual system (dark/light aware)
data/timeline.json    structured source-of-truth for every citation (order, name, date, source)
js/helpers.js         shared canvas/slider/toggle utilities used by every widget
js/nav.js             builds the left-hand timeline rail and scroll-spy highlighting
js/main.js             wires every widget to its container on page load
js/widgets/*.js       one module per interactive widget, several reused across sections
                      with different parameters (head-sharing for MQA+GQA, sliding-window
                      for Longformer+Mistral, delta-rule for the three delta-rule sections,
                      rope-scaling for PI+NTK+YaRN, top-k for NSA+DSA)
```

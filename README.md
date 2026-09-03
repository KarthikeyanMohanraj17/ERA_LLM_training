# How Attention Actually Got Here

An interactive, chronological tour of every major attention mechanism, from scaled
dot-product attention in 2017 to the hybrid stacks of 2026. Static site — no framework, no
backend, nothing to compile in order to view it.

**Live:** <https://unrivaled-chimera-0d8a5a.netlify.app/> · **Sources and date receipts:**
<https://unrivaled-chimera-0d8a5a.netlify.app/sources>

**Repository:** <https://github.com/KarthikeyanMohanraj17/ERA_LLM_training> — this project
lives in `session8_attention-timeline/`.

## What this sets out to do

Standard attention was never wrong. It solved a real problem and sent two bills: compute
growing as T², and a KV cache growing with every generated token. Everything after it is
the field deciding, workload by workload, which of those bills to pay.

So the page is built as a chain rather than a catalogue. Every mechanism section walks the
same chain in order:

```
what existed  →  what problem people hit  →  what mechanism someone proposed
              →  what it fixed  →  what new trade-off it introduced
              →  and who had to fix THAT
```

That last link is the one that turns a dated list into a story, so every mechanism section
carries it explicitly — either naming the later mechanism that attacks its cost, or saying plainly that
the cost was never resolved and the thread went quiet. Four sections end in **Still open** — ALiBi,
NoPE, attention sinks and DeepSeek Sparse Attention — and they say so rather than trailing off.

Three things follow from that framing:

- **Sections are ordered by real launch date**, not by teaching order or topic. That means
  threads interleave: sparsity appears in 2019, goes quiet for five years and returns in
  2025; recurrence appears in 2020 and comes back in 2024. Six **era bands** mark what the
  field was mostly working on at each point, each stating the bill that forced it, what got
  cheaper, and what it handed on: *Make it work at all* (2017–18) → *The first bills come
  due* (2019–20) → *Position gets rebuilt, and one bill turns out to be fake* (2021–22) →
  *Stretch the context* (2023) → *Recurrent state and sparsity both return* (2024–25) →
  *Compression gets aggressive* (2025–26).
- **Every mechanism is a trade.** Each section answers what it buys, what it gives up, and
  when you would actually choose it — the last being the one that matters, because a
  mechanism can be right for a 2K chatbot and wrong for a 1M-token agent. No section claims
  a free lunch; where a cost is small, the section says how small and why.
- **Every date is checked against the primary source.** That is the part easiest to get
  wrong and easiest to check, so *How every date was verified* below gives the method, a
  command you can re-run, and a row per paper.

## What it covers

28 sections: 26 mechanisms, a closing case study that assembles several of them into one
stack, and a KV-cache calculator for the cost formula the memory sections all share.

The 26, in page order: standard attention, sinusoidal positions, relative position, learned
absolute positions, sparse attention, MQA, sliding window, linear attention, the delta rule,
RoPE, ALiBi, FlashAttention, GQA, NoPE, Position Interpolation, NTK-aware scaling, YaRN,
decoder sliding window, attention sinks, MLA, Mamba-2, DeltaNet, Gated DeltaNet, NSA,
DeepSeek Sparse Attention (top-k selection, over compressed blocks and over individual
tokens respectively), and DroPE.

Each section has a **worked example**: a hand-computable dry run in Given → steps → Answer
form, using numbers small enough to check on paper. One query over two keys. The exact bytes
MQA saves per token. The arithmetic showing linear attention's regrouping trick returns the
same answer twice. The 51.54 GB a KV cache actually costs at 32K context and 8 users.

25 interactive widgets, each built so the interaction teaches the mechanism rather than
decorating it — toggle softmax on and watch linear attention's shortcut break; shift both
RoPE positions by ten and watch the score refuse to move.

## Running it locally

```
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## How every date was verified

**The rule.** Each date is the arXiv **v1 submission date** — the first time the work was public —
not the v2 date, not the conference date, not the date a blog post described it later. Where the
primary source is not a preprint, the rule is the official release date, and those four entries are
listed separately below with their own sourcing.

**The method.** Every arXiv ID cited anywhere on the page was extracted from `index.html` and sent
to the arXiv API in one query; the `<published>` field it returns is the v1 timestamp. Titles and
author lists came back in the same response and were diffed against what the page claims, because a
half-remembered *title* is the same class of error as a half-remembered date.

**Re-run it yourself.** This needs nothing but `curl` and Python, takes a few seconds, and prints a
row per paper:

```bash
IDS=$(grep -o 'arxiv\.org/abs/[0-9.]*' index.html | cut -d/ -f3 | sort -u | paste -sd, -)
curl -sG https://export.arxiv.org/api/query \
  --data-urlencode "id_list=$IDS" --data-urlencode "max_results=60" |
python3 -c '
import sys,xml.etree.ElementTree as ET
ns={"a":"http://www.w3.org/2005/Atom"}
for e in ET.parse(sys.stdin).getroot().findall("a:entry",ns):
    print(e.find("a:published",ns).text[:10],
          e.find("a:id",ns).text.rsplit("/",1)[-1],
          " ".join(e.find("a:title",ns).text.split())[:60])'
```

**Result of the last run: 33 of 33 IDs resolved, and every arXiv-dated section matches its paper's
v1 date.** (33 rather than 24, because several sections cite context papers alongside their primary
source — Transformer-XL and T5 under relative position, Haviv et al. under NoPE, and so on.) The same pass returns titles and author lists, which are diffed against the page, so credits
follow the record — FlashAttention-2 and -3 carry their own author lines, because FA-3's first
author (Jay Shah) is not FA-1's.

| Section | Date on the page | arXiv | v1 per arXiv | Title as recorded |
|---|---|---|---|---|
| Standard attention | Jun 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) | **12 Jun 2017** | Attention Is All You Need |
| Sinusoidal PE | Jun 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) | **12 Jun 2017** | Attention Is All You Need |
| Relative position | Mar 2018 | [1803.02155](https://arxiv.org/abs/1803.02155) | **6 Mar 2018** | Self-Attention with Relative Position Representations |
| Sparse Transformers | Apr 2019 | [1904.10509](https://arxiv.org/abs/1904.10509) | **23 Apr 2019** | Generating Long Sequences with Sparse Transformers |
| MQA | Nov 2019 | [1911.02150](https://arxiv.org/abs/1911.02150) | **6 Nov 2019** | Fast Transformer Decoding: One Write-Head is All You Need |
| Sliding window (origin) | Apr 2020 | [2004.05150](https://arxiv.org/abs/2004.05150) | **10 Apr 2020** | Longformer: The Long-Document Transformer |
| Linear attention | Jun 2020 | [2006.16236](https://arxiv.org/abs/2006.16236) | **29 Jun 2020** | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention |
| The delta rule | Feb 2021 | [2102.11174](https://arxiv.org/abs/2102.11174) | **22 Feb 2021** | Linear Transformers Are Secretly Fast Weight Programmers |
| RoPE | Apr 2021 | [2104.09864](https://arxiv.org/abs/2104.09864) | **20 Apr 2021** | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| ALiBi | Aug 2021 | [2108.12409](https://arxiv.org/abs/2108.12409) | **27 Aug 2021** | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation |
| FlashAttention | May 2022 | [2205.14135](https://arxiv.org/abs/2205.14135) | **27 May 2022** | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| GQA | May 2023 | [2305.13245](https://arxiv.org/abs/2305.13245) | **22 May 2023** | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| NoPE | May 2023 | [2305.19466](https://arxiv.org/abs/2305.19466) | **31 May 2023** | The Impact of Positional Encoding on Length Generalization in Transformers |
| Position Interpolation (bonus) | Jun 2023 | [2306.15595](https://arxiv.org/abs/2306.15595) | **27 Jun 2023** | Extending Context Window of Large Language Models via Positional Interpolation |
| YaRN | Aug 2023 | [2309.00071](https://arxiv.org/abs/2309.00071) | **31 Aug 2023** | YaRN: Efficient Context Window Extension of Large Language Models |
| Sliding window (decoder) | Sep–Oct 2023 | [2310.06825](https://arxiv.org/abs/2310.06825) | **10 Oct 2023** | Mistral 7B |
| Attention sinks | Sep 2023 | [2309.17453](https://arxiv.org/abs/2309.17453) | **29 Sep 2023** | Efficient Streaming Language Models with Attention Sinks |
| MLA | May 2024 | [2405.04434](https://arxiv.org/abs/2405.04434) | **7 May 2024** | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model |
| Mamba-2 | May 2024 | [2405.21060](https://arxiv.org/abs/2405.21060) | **31 May 2024** | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |
| DeltaNet (parallel training) | Jun 2024 | [2406.06484](https://arxiv.org/abs/2406.06484) | **10 Jun 2024** | Parallelizing Linear Transformers with the Delta Rule over Sequence Length |
| Gated DeltaNet | Dec 2024 | [2412.06464](https://arxiv.org/abs/2412.06464) | **9 Dec 2024** | Gated Delta Networks: Improving Mamba2 with Delta Rule |
| Native Sparse Attention | Feb 2025 | [2502.11089](https://arxiv.org/abs/2502.11089) | **16 Feb 2025** | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention |
| DroPE | Dec 2025 | [2512.12167](https://arxiv.org/abs/2512.12167) | **13 Dec 2025** | Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings |
| LightningLM 0.1V (case study) | Jun 2026 | [2606.07404](https://arxiv.org/abs/2606.07404) | **5 Jun 2026** | Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling |

### The four entries with no arXiv paper

| Section | Date on the page | Source, and how solid it is |
|---|---|---|
| Learned PE | Jun 2018 | GPT-1 has no arXiv entry; the date is OpenAI's [paper PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) release. The date is GPT-1's, but the mechanism is older — see *Notes on the entries that are harder to source* below. |
| NTK-aware scaling | ~Jun 30, 2023 | An r/LocalLLaMA post by u/bloc97, not a paper. Reddit's own timestamp was not directly fetchable, so this date is corroborated by secondary sources only. The page marks it approximate with a `~` and the section says it is not peer reviewed. **This is the weakest date on the page and is labelled as such.** |
| DSA | Sep 2025 | [DeepSeek-V3.2-Exp release notes](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) on GitHub, 29 Sep 2025. A dated repository release, not a preprint. |
| KV-cache calculator | — | Not a mechanism and carries no date; it is a reference tool for the shared cost formula. |

### What "verified" does and does not mean here

It means: the paper exists, the ID resolves, the v1 date matches, and the title and first author on
the page match the record. It does **not** mean every claim about every mechanism was checked
against the full text. Where a specific claim *was* checked against full text — the LightningLM
numbers, the Vaswani learned-PE quote, the DroPE abstract — the section itself says so.
Where it was not, treat the description as a careful reading of the abstract.

## Sources for every date used in the chronology

The table above is the machine-checked receipt — arXiv API output for the IDs as cited. This
one is the human-readable chronology with full author lists, and the only place the non-arXiv
sources appear in full.


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
| 11 | Position Interpolation | Chen, Wong, Chen, Tian (Meta), "Extending Context Window of Large Language Models via Positional Interpolation," [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) | Jun 2023 |
| 12 | NTK-aware RoPE scaling | u/bloc97, r/LocalLLaMA community post (not peer-reviewed); related independent work by kaiokendev the same month | ~Jun 30, 2023 (approximate — see note below) |
| 13 | YaRN | Peng, Quesnelle, Fan, Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | Aug 2023 |
| 14 | Sliding window attention (decoder + KV-cache application) | Mistral AI, [official blog](https://mistral.ai/news/announcing-mistral-7b/); Jiang et al., "Mistral 7B," [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) | Sep 2023 (blog) / Oct 2023 (paper) |
| 15 | Attention sinks / StreamingLLM | Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks," [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | Sep 2023 |
| 16 | Multi-head Latent Attention (MLA) | DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) | May 2024 |
| 17 | DeltaNet (parallel-training algorithm for the delta rule) | Yang, Wang, Zhang, Shen, Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) | Jun 2024 |
| 18 | Gated DeltaNet | Yang, Kautz, Hatamizadeh (NVIDIA), "Gated Delta Networks: Improving Mamba2 with Delta Rule," [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | Dec 2024 |
| 19 | Native Sparse Attention (NSA) | Yuan, Gao, Dai, Luo, Zhao, Zhang et al. (DeepSeek-AI, Peking University, University of Washington), "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention," [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | Feb 2025 |
| 20 | DeepSeek Sparse Attention (DSA) | DeepSeek-AI, [DeepSeek-V3.2-Exp release](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | Sep 2025 |
| 21 | DroPE | Gelberg, Eguchi, Akiba, Cetin (Sakana AI), "DroPE: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings," [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) | Dec 2025 |
| 22 | LightningLM 0.1V (case study) | Shravan (The School of AI), "Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling," [arXiv:2606.07404](https://arxiv.org/abs/2606.07404) | Jun 2026 |

## Mechanisms beyond the usual list

Each was held to the same standard as the rest of the timeline: v1 date fetched from the arXiv
API, title and author list verified, and a stated motivation / mechanism / advantage / cost /
timeline slot.

| Page position | Mechanism | Source | v1 date (verified) | Why it belongs |
|---|---|---|---|---|
| 02 | Relative position representations | Shaw, Uszkoreit, Vaswani — [arXiv:1803.02155](https://arxiv.org/abs/1803.02155); refined by Transformer-XL [arXiv:1901.02860](https://arxiv.org/abs/1901.02860), simplified by T5 [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) | 6 Mar 2018 | RoPE and ALiBi are both relative schemes, and without this the timeline gives them no ancestor. ALiBi in particular *is* T5's scalar logit bias with the learned value replaced by a fixed slope — unreadable as an invention out of nowhere. |
| 11 | FlashAttention | Dao, Fu, Ermon, Rudra, Ré — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | 27 May 2022 | The largest gap in the core sequence: 21 months with nothing in it. It is also the page's one counterexample — exact attention made dramatically cheaper at zero quality cost — which reframes every approximation around it. Much of the "quadratic cost" being approximated away was memory traffic, and that part had an exact answer. |
| 13 | NoPE | Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy — [arXiv:2305.19466](https://arxiv.org/abs/2305.19466); earlier evidence Haviv et al. [arXiv:2203.16634](https://arxiv.org/abs/2203.16634) | 31 May 2023 | The control condition for six sections of position engineering, and DroPE's direct ancestor — cited in DroPE's own paper. |
| 20 | Mamba-2 | Dao, Gu — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060); predecessor Mamba [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | 31 May 2024 | Gated DeltaNet's literal title is *"Improving Mamba2 with Delta Rule"*. Without this section that title cannot be read, and the convergence of the state-space and linear-attention branches is invisible. |

## Notes on the entries that are harder to source

Four entries do not resolve to a single clean preprint, and each is flagged in the app itself
rather than smoothed over.

- **Absolute learned positional embeddings (Jun 2018).** The date is GPT-1's, but the
  mechanism is older, and the usual "sinusoidal was limiting, so someone invented learned
  embeddings" story is not what happened. Both options are in the 2017 Transformer paper,
  tested against each other — §3.5: *"We also experimented with using learned positional
  embeddings instead, and found that the two versions produced nearly identical results."*
  They chose sinusoidal on a hypothesis about extrapolation, not a measurement. The learned
  table itself goes back further still, to Gehring et al.
  ([arXiv:1611.02344](https://arxiv.org/abs/1611.02344), Nov 2016), predating the Transformer.
  The section is dated by its cited source and explains this in full.

- **NTK-aware scaling (~Jun 30, 2023).** A Reddit post, not a paper, and the weakest date
  here. Reddit's own timestamp was not directly fetchable, so the date rests on secondary
  corroboration. Marked approximate with a `~` throughout, and the section says it was never
  peer reviewed.

- **The delta rule vs. "DeltaNet".** Routinely conflated. The delta rule is Schlag, Irie and
  Schmidhuber, Feb 2021. "DeltaNet" — what people usually mean — is Yang et al., Jun 2024,
  which is a parallel-*training* algorithm for that same recurrence, not the origin of the
  idea. Both get their own entry so the distinction is visible rather than asserted.

- **Sliding window attention has two distinct contributors.** Longformer (Apr 2020)
  originated it for encoder-style long documents with global tokens. Mistral 7B (Sep 2023)
  applied the same core idea inside causal, KV-cached decoding with a rotating buffer. Both
  get separate entries, cross-referenced.

### A caution on the LightningLM case study

The closing case study is a real, citable paper
([arXiv:2606.07404](https://arxiv.org/abs/2606.07404), v1 5 Jun 2026), and its architecture
claims check out against its own text — the `DDDGDDDG` motif is §3.1/Table 1, the Memory
Stream is §3.2, DroPE before annealing is §5.3. Two things widely said about it do not.

The **"8K → 256K, 32× context extension"** figure is not in the paper. Searching its full
text returns zero hits for `256K` and zero for `32×`, and it states the opposite outright:

> "The production models here were trained and evaluated at 4K and 8K, the longer reaches
> were not tested in this work."

The 8× and 80× extraction factors it does cite belong to Gelberg et al. 2025 — the DroPE
authors — which from an 8K base would be roughly 64K and 640K. Neither is 256K.

There is also **no D:G ratio sweep** anywhere in the paper. Table 5 holds the ratio at 3:1
for every stage it trains: 6 D : 2 G at eight layers, 15 D : 5 G at twenty. So any
cache-versus-attention-layer curve attributed to it is not a measurement. The widget in that
section shows only what is derivable — if only G layers hold a cache, cache share is
`G ÷ 8` — and labels it as arithmetic rather than a result.

### DroPE's ancestry

DroPE is presented as the answer to a question first asked earlier: whether the positional
signal is worth keeping at all. Haviv et al. asked it in
[Mar 2022](https://arxiv.org/abs/2203.16634) and Kazemnejad et al. answered it at NeurIPS
2023 ([arXiv:2305.19466](https://arxiv.org/abs/2305.19466)), finding that NoPE beat ALiBi,
RoPE and T5-relative on length generalization. DroPE's own paper cites both. NoPE therefore
gets its own section, and DroPE is framed as the fix for NoPE's real weakness — train *with*
positions, then remove them — rather than as an idea from nowhere.

Note also that the arXiv title carries no `DroPE:` prefix; "DroPE" is a backronym coined
inside the abstract.

## Illustrative vs. measured numbers

Almost every widget computes from the published rule for its mechanism, or from arithmetic
you can redo by hand. The RoPE-scaling trio is derived, not drawn: Position Interpolation is
`1/s` applied uniformly, NTK-aware is `1/(1 + (s−1)·d)`, and YaRN implements the published
NTK-by-parts rule, keyed on each dimension's wavelength against the trained context.

Exactly two widgets are not derived, and both say so on their own canvas. DroPE's
quality-vs-length pair is labelled *"Illustrative shape, not measured perplexity"*, because
the paper's claim is directional and this shows the direction. The LightningLM depth-schedule
bar is labelled *"Arithmetic, not a measurement"*, because cache share follows from the layer
count and nothing in that paper measures it.

The KV-cache calculator computes an exact, checkable formula:

```
bytes = 2 × layers × kv_heads × head_dim × context × batch × bytes_per_number
```

Its defaults (48 layers, 8 KV heads, head_dim 128, bf16, 32,768-token context) give
≈6.44 GB for one user and ≈51.54 GB at 8 concurrent users — worked through by hand in the
section itself.

## Structure

```
index.html            single scrolling page, one section per mechanism, chronological
styles.css            shared visual system (dark/light aware)
sources.html          this README, rendered for the live site (generated — do not hand-edit)
build-sources.py      regenerates sources.html from README.md
netlify.toml          publish dir and cache headers
_headers              the same headers again, for drag-and-drop deploys, which ignore netlify.toml
404.html              not-found page
js/helpers.js         canvas setup, chart()/legend()/caption()/tokenAxis() primitives,
                      font(), slider and toggle bindings, softmax
js/nav.js             builds the left-hand timeline rail and scroll-spy highlighting
js/main.js            wires every widget to its container on page load
js/widgets/*.js       one module per interactive widget, several reused across sections
                      with different parameters (head-sharing for MQA+GQA, sliding-window
                      for Longformer+Mistral, delta-rule for delta-rule-origin+Gated DeltaNet,
                      rope-scaling for PI+NTK+YaRN, top-k for NSA+DSA — with NSA additionally
                      showing its three parallel branches). alibi.js, drope.js and
                      delta-parallel.js are each dedicated to a single section.
                      delta-parallel.js is separate from delta-rule.js because DeltaNet's
                      contribution is training speed, not new correction maths — it charts
                      sequential vs chunk-parallel steps rather than repeating the rule.
```

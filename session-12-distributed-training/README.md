# Where the memory goes when you train on 32 GPUs

Data parallelism does not run out of memory because the model is large. It runs out because
every GPU holds a bit-identical copy of optimizer state it will never independently use. ZeRO
is the observation that you can shard that copy and reassemble it only when it is needed — and
that for two of its three stages, doing so costs **no extra communication at all**.

**Notebook:** [`zero_32_virtual_gpus.ipynb`](zero_32_virtual_gpus.ipynb) ·
**Measurements:** [`results/metrics.csv`](results/metrics.csv),
[`results/summary.json`](results/summary.json) ·
**Repository:** https://github.com/KarthikeyanMohanraj17/ERA_LLM_training — this project lives
in `session-12-distributed-training/`.

Thirty-two virtual GPUs built from CPU threads, each with a byte-accurate memory ledger and a
hand-written ring collective fabric. A small transformer trains on top of them in four modes —
plain data parallelism, then ZeRO stages 1, 2 and 3 — and the notebook measures what changed.

## What this sets out to do

Anyone can write a script that prints plausible memory numbers for ZeRO. The interesting
question is whether the numbers are *checkable*, and the whole project is arranged around
making them so. Three commitments, stated before any evidence:

**Every number is measured or derived, and labelled which.** Memory comes from an arena that
counts bytes as they are allocated and freed. Communication comes from counters inside the
collectives. Where a closed form exists, the measured value is asserted equal to it. Where
neither is available — communication *time* — the number is labelled a model and nothing
depends on it.

**Every correctness claim has a matching bug that makes it fail.** Four deliberate defects are
in the repository, each one something a person would plausibly ship. All four still produce a
falling loss curve. If the assertions could not catch them, the assertions would not be
evidence.

**The dishonest parts are enumerated, not hedged.** There is a section called
[What this does not prove](#what-this-does-not-prove), it is placed before the results rather
than after them, and it names mechanisms rather than gesturing at limitations.

The chain the rest of this document follows:

```
what is replicated  ->  what you shard  ->  what per-GPU memory becomes
                    ->  what extra collective that forces
                    ->  what the collective costs  ->  what you gave up
```

## Why data parallelism hits a wall

Start with one card and count bytes by hand.

GPT-2 XL has 1.5 billion parameters. Put it on a 16 GB card and train it with Adam in mixed
precision. Resident at all times: 3 GB of fp16 weights, 3 GB of fp16 gradients, a 6 GB fp32
master copy of the weights (fp16 updates lose too much precision to be applied directly), 6 GB
for Adam's first moment, and 6 GB for its second. **24 GB, before a single activation exists.**
The ZeRO paper says it plainly: *"a 1.5B parameter GPT-2 model requires 3GB of memory for its
weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with
32GB memory."*

The part that makes ZeRO inevitable is not the total. It is that 18 of those 24 GB are
optimizer state, and that state is **identical on every rank** — every GPU reduces to the same
averaged gradient and therefore computes the same update. A 33rd GPU does not shrink it by one
byte. Data parallelism scales throughput and does exactly nothing for memory.

Written out, for Ψ parameters:

| Component | mixed precision (the paper) | fp32 + Adam (what I run) |
|---|---|---|
| weights used in the forward pass | 2Ψ (fp16) | 4Ψ (fp32) |
| gradients | 2Ψ (fp16) | 4Ψ (fp32) |
| fp32 master copy of the weights | 4Ψ | — (already fp32) |
| Adam first moment `m` | 4Ψ | 4Ψ |
| Adam second moment `v` | 4Ψ | 4Ψ |
| **total per GPU** | **16Ψ** | **16Ψ** |

The paper writes the middle column as 2Ψ + 2Ψ + KΨ with K = 12.

**Both columns come to 16 bytes per parameter, and that coincidence is a trap.** It makes the
two regimes look interchangeable. They are not, because ZeRO stage 1 shards only the optimizer
state — which is 8/16 of the total in fp32 and 12/16 in mixed precision. The *same* stage 1
therefore saves 1.94× in one regime and 3.66× in the other. My simulator runs fp32, so my
measured numbers are the weaker column, and I report them as mine rather than quoting the
paper's next to them.

## The three stages

Sizes below are from this repository's model: Ψ = 269,821 parameters, N = 32,
`blocks.2.mlp.w1` is 16,384 elements sitting in a 50,176-element group.

### ZeRO-1 — shard the optimizer state

**What is sharded.** Adam's `m` and `v`. Each rank keeps the whole parameter tensor and the
whole gradient, but only 1/32 of the momentum and variance. Of `block2`'s 50,176-element flat
buffer, rank 7 owns Adam state for elements 10,976 through 12,543 and nothing else.

**Per-GPU memory.** 4Ψ + 4Ψ + 8Ψ/N. At N = 32 that is 8.25Ψ, so 1.94× less than DDP in fp32;
in the paper's mixed precision it is 4Ψ + 12Ψ/N, which is 3.66× less.

**Communication.** `reduce_scatter` the gradients, run Adam on your own slice,
`all_gather` the updated parameters. Measured: **2,103,040 bytes per rank per step —
identical to DDP, to the byte.** The next section is why.

**What you give up.** The parameter all-gather now sits on the critical path *after* the
optimizer step, where DDP had nothing. In a real implementation that is harder to overlap with
compute, because there is no forward work left to hide it behind.

### ZeRO-2 — and the gradients

**What is sharded.** The same, plus gradients. This is the stage where implementation matters
most, because ZeRO-2 runs *exactly the same collectives on exactly the same payloads* as
ZeRO-1. The only difference is when the gradient buffer is allowed to exist.

The naive version reduce-scatters after backward completes — at which point the full Ψ-sized
gradient buffer has already existed, and ZeRO-2 measures identically to ZeRO-1. To get the
memory back you have to free each group's gradient *during* backward, the moment it is
complete. In this repository that is a `register_post_accumulate_grad_hook` per group which
reduce-scatters and then sets `.grad = None`
([`vzero/engine.py`](vzero/engine.py), `Worker._install_grad_hooks`).

**Per-GPU memory.** 4Ψ + 4Ψ/N + 8Ψ/N = 4.375Ψ at N = 32, or 2Ψ + 14Ψ/N in mixed precision.
Measured: 1,187,200 bytes of model state against ZeRO-1's 2,272,640 — a 1.91× step for free.

**Communication.** Unchanged. Still 2,103,040 bytes per rank per step.

**What you give up.** No rank holds a whole gradient any more, so anything that needs one
globally now needs its own collective. Gradient-norm clipping is the common case: `clip_grad_norm_`
computes a global norm across all parameters, and under ZeRO-2 that becomes a partial sum per
rank plus an all-reduce. Miss it and clipping silently uses the wrong norm.

### ZeRO-3 — and the parameters

**What is sharded.** Everything. A rank holds 1/32 of each parameter and materialises a group's
full weights only for as long as it takes to use them.

**Per-GPU memory.** 16Ψ/N — measured 135,680 bytes against DDP's 4,341,760, which is
**exactly 32.00×**. Plus a transient buffer, which is not a footnote; see below.

**Communication.** `all_gather` the parameters in forward, free them, `all_gather` them again in
backward, `reduce_scatter` the gradients. Three phases where the others used two. Measured:
3,154,560 bytes per rank per step = **1.5000× DDP**.

**What you give up.** No rank holds a whole parameter, so `state_dict()`, checkpointing, and
every debugger breakpoint become collective operations. This project hits it directly:
reassembling the model for comparison against the reference has to concatenate 32 shards rather
than read one rank (`assemble_params` in [`vzero/engine.py`](vzero/engine.py)).

| Stage | sharded | per-GPU (fp32) | at N=32 | measured state | comm/step | measured |
|---|---|---|---|---|---|---|
| ZeRO-0 | nothing | 16Ψ | 16.00Ψ | 4,341,760 B | 2(N−1)/N·Ψ·4 | 2,103,040 B |
| ZeRO-1 | optimizer | 4Ψ+4Ψ+8Ψ/N | 8.25Ψ | 2,272,640 B | same | 2,103,040 B |
| ZeRO-2 | +gradients | 4Ψ+4Ψ/N+8Ψ/N | 4.375Ψ | 1,187,200 B | same | 2,103,040 B |
| ZeRO-3 | +parameters | 16Ψ/N | 0.50Ψ | 135,680 B | 3(N−1)/N·Ψ·4 | 3,154,560 B |

ZeRO-1's measured 2,272,640 is 33,920 bytes above the formula's 2,238,720. That is not slop:
the reduce-scatter writes into a separate shard-sized gradient buffer which is resident
alongside the full one, and 33,920 = 4 × 8,480 is exactly one shard. Small, but it is the
difference between a number I can account for and a number I cannot.

## Why ZeRO-1 and ZeRO-2 cost exactly what DDP costs

This is the most commonly misunderstood thing about ZeRO, and I got it wrong before I counted.

The wrong intuition is reasonable: DDP does one collective, ZeRO-1 does two, so ZeRO-1 must
cost more. The error is in the first clause. **A ring all-reduce is already two collectives.**
It is a reduce-scatter followed by an all-gather — not as an approximation, but as its actual
implementation:

```
reduce-scatter   N-1 steps. Each rank sends one chunk to its right neighbour, which
                 adds its own contribution. After N-1 steps rank r holds the complete
                 sum of chunk r, and no rank holds the whole reduced tensor.

all-gather       N-1 steps. Each rank passes the chunk it owns around the ring until
                 every rank has all N chunks.
```

Each phase moves (N−1)/N·Ψ elements per rank, so the pair moves 2(N−1)/N·Ψ. That is DDP's cost,
and it is also ZeRO-1's cost, because ZeRO-1 does the same two phases on payloads of the same
size. Nothing was added. What changed is *what the all-gather carries*: DDP's carries reduced
gradients back to everyone, ZeRO-1's carries updated parameters. Same element count, different
tensor. The optimizer step simply moved from after both phases to between them.

ZeRO-2 changes neither collective — it only frees the gradient earlier — so it is free too.

The receipt, from [`results/metrics.csv`](results/metrics.csv): **2,103,040 bytes per rank per
step for ZeRO-0, ZeRO-1 and ZeRO-2 alike.** Not approximately equal. The same integer. And the
byte counters are asserted against the closed form 2(N−1)/N·Ψ_padded·4 with no tolerance,
because for integer counters over a deterministic schedule any discrepancy is a real extra
message rather than noise.

The corollary most write-ups skip: if ZeRO-2 costs what DDP costs and uses 3.66× less memory,
there is no good reason to run plain DDP when ZeRO-2 is available.

**And a footgun I built on purpose, because I nearly shipped it.** ZeRO-1 does not have to be
written that way. The tempting implementation is "all-reduce the gradients, then have each rank
update its own shard, then all-gather the parameters." It is *correct*. It produces a
**bitwise identical** model. It uses the same memory. And it costs 3,154,560 bytes per step —
1.5× DDP — for nothing at all, because the all-reduce already broadcast gradients that were then
thrown away. It is in the repo as `zero1_naive=True` and it is the grey bar in
`figures/03-comm-volume.png`. A correct loss curve would never have told me.

## Why ZeRO-3 is 1.5×, and why you take the deal

The third phase is easy to locate: the parameters gathered for the forward pass are released
before backward, so they have to be gathered again. Three phases against two:
3(N−1)/N ÷ 2(N−1)/N = 1.5 exactly, for every N — not just at 32. Measured 1.5000.

Whether that is worth it is a trade with numbers on both sides. Fifty percent more
communication buys 8.75× less model state than ZeRO-2 at N = 32. Communication is the cheaper
side because compute per step scales with batch × sequence while this communication scales only
with Ψ — so the memory ZeRO-3 frees can be spent on a larger batch, which buys back the ratio.

**But 1.5× is a default, not a law.** Keep the gathered parameters resident between forward and
backward and the second gather disappears: communication drops to 2(N−1)/N — DDP's volume — at
the cost of holding one unsharded group longer. That is `reshard_after_forward` in PyTorch
FSDP, and it is the difference between FSDP's `FULL_SHARD` and `SHARD_GRAD_OP`. Going the other
way, adding activation checkpointing means the recomputed forward needs a *third* gather,
pushing past 3Ψ. Quoting 1.5× as a property of ZeRO-3 rather than of one configuration is a
mistake.

## What this does not prove

Placed here, before the results, because it changes how they should be read.

**Threads are not GPUs.** Thirty-two Python threads share one address space and one interpreter
lock. "Per-GPU memory" here is a ledger maintained in software and cross-checked against summed
storage sizes. It does **not** mean a measurement of any device's memory.

**There is no network.** A collective is a memcpy between Python objects. The byte counts are
real, because the ring is genuinely executed chunk by chunk and the counters match the closed
form exactly — a real 31-step ring implementation is in the repo and is asserted to produce
byte-identical counts to the fast path. The *times* are not real, because there is no wire.

**No speed claim is made anywhere.** Under the GIL, 32 threads on 12 cores measure the
simulator, not ZeRO. There is no throughput number, no speedup plot, and no wall-clock figure
in any chart here. That is deliberate rather than an oversight: a caveated number gets
screenshotted without its caveat.

**fp32, not mixed precision.** The measured ratios are the weaker column of the table above.
The paper's headline numbers are reproduced analytically, not measured.

**Ψ = 269,821, not 269 billion.** Five orders of magnitude below anything real. Every quantity
linear in Ψ carries over as a ratio; two do not, and both are flagged where they appear — the
transient gather buffer and activation memory.

**One world size is measured end to end.** N = 32 everywhere except the sweep in
`figures/04-memory-vs-N.png`.

What it *does* prove: that the analytical formulas are right — they reproduce the paper's
Table 1 exactly at both 7.5B and 128B; that a from-scratch implementation of each stage is
**bitwise identical** to a single-rank reference; that the communication volumes are what the
arithmetic says, to the byte; and that the assertions making those claims can fail, because
four deliberate bugs are in the repository and every one of them still produced a falling loss
curve.

## What the runs show

Thirty-two virtual GPUs, Ψ = 269,821, 12 steps, fp32 + Adam. All 32 validation checks pass;
the full list with measured values is in
[`results/summary.json`](results/summary.json).

### Memory

![memory breakdown](figures/01-memory-breakdown.png)

Each stage removes exactly the component it claims to. What the bar chart shows that the table
does not is the red segment: **it is identical in all four stages.** Activations do not shard
under any ZeRO stage, because ZeRO partitions model state and activations are not model state.
Measured at 1,495,428 bytes in every mode, and identical on all 32 ranks — both asserted rather
than eyeballed.

The consequence is the gap between two columns: ZeRO-3 reduces *model state* by 32.00× and
*peak memory* by 3.33×, because on a model this small the activations it cannot touch are most
of what is left. That is the real situation, not a simulation artifact, and it is why
activation checkpointing is not optional at scale. It is a different axis, and it is not ZeRO.

The bars are stacked using each bucket's size **at the instant of peak**, not each bucket's own
peak. Those differ — buckets do not all peak together — and summing per-bucket peaks overstated
ZeRO-2 by 200,704 bytes and ZeRO-3 by 281,860 before I noticed.

![measured vs analytical](figures/02-measured-vs-analytical.png)

### Communication

![communication volume](figures/03-comm-volume.png)

Three bars at exactly the same height, one at exactly 1.5×, and the grey footgun. Black ticks
are the closed form.

### Scaling

![memory vs N](figures/04-memory-vs-N.png)

ZeRO-1 and ZeRO-2 flatten out. They have a floor set by what they do not shard — 8Ψ and 4Ψ in
fp32 with Adam — and no number of GPUs gets below it. Only ZeRO-3 has no floor. That asymptote,
not the value at any particular N, is the argument for stage 3.

![padding](figures/06-padding.png)

![scaling](figures/07-scaling.png)

Largest model that fits 80 GiB per GPU, 25% reserved for activations, mixed precision, 32 layers:

| N | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 8 | 4.0B | 11.7B | 17.2B | 29.5B |
| 32 | 4.0B | 14.7B | 26.4B | 93.7B |
| 64 | 4.0B | 15.4B | 29.0B | 147.3B |
| 512 | 4.0B | 16.0B | 31.8B | 294.5B |

The activation reserve is an assumption, stated as a visible parameter rather than a footnote,
because with it set to zero every cell becomes a weights-only upper bound that is not trainable.

### Correctness

![loss and ablations](figures/05-loss-and-ablations.png)

| variant | max \|ΔLoss\| vs reference | loss still fell? | caught by |
|---|---|---|---|
| correct ZeRO-0/1/2/3 | **0.000e+00** | — | — |
| `no_reduce` | 1.610e-01 | yes | a 1e-4 tolerance |
| `shard_offset` | 1.377e-01 | yes | a 1e-4 tolerance |
| `no_all_gather` | 2.045e-01 | yes | a 1e-4 tolerance |
| `sum_not_mean` | **5.007e-06** | yes | **only the bitwise check** |

Every broken variant still trains. "The loss went down" is not evidence of anything.

## Things that got me

**The memory ledger was wrong in a direction nothing could catch.** Counting `numel × itemsize`
per tensor double-counts every `.view()`, because a view shares storage with its base. Keying
on `untyped_storage().data_ptr()` fixes it. Before that, every memory number was inflated — and
no assertion would have noticed, because all four modes were inflated together. The lesson is
that a ledger nothing can falsify is not evidence.

**Then the fix had its own bug.** Keying on an address without holding a reference to the
storage means the storage gets freed, the allocator hands the same address to the next tensor,
and the ledger concludes it is "already counted." From then on it undercounts. The arena now
keeps a strong reference to each storage it charges, which makes address reuse impossible while
counted and turns a forgotten `free()` into a detectable leak instead of a wrong number. The
test that caught it is four lines and calls `gc.collect()` between two allocations.

**Autograd will not let you free a gathered weight.** ZeRO-3's entire premise is that a rank
holds 1/32 of a parameter and materialises the rest only briefly. But `F.linear` saves its
weight for backward, so dropping your reference frees nothing. Real FSDP resizes the underlying
storage to **zero bytes** and resizes it back before backward needs it — saved views follow the
storage. That is what `GroupState.reshard` does, and it is why ZeRO-3 here can use ordinary
`F.linear`, `F.embedding` and `F.layer_norm` instead of hand-written backward passes, which in
turn is why it comes out bitwise identical to the other modes rather than merely close.

**Writing the re-gathered weights then trips autograd's version counter.** The gather in
backward writes into the same storage whose views autograd saved during forward, so it raises
*"one of the variables needed for gradient computation has been modified by an inplace
operation."* The escape hatch is `torch.autograd._unsafe_preserve_version_counter`, which is
not a hack I invented — FSDP2 wraps its own all-gather copy-out in the same context manager, in
`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`. I found the fix by reading how
FSDP solves the same problem.

**Adam is scale-invariant, so the classic bug is nearly invisible.** Forgetting to divide the
reduction by N makes every gradient 32× too large. Under SGD that is a 32× learning rate and an
immediate blow-up. Under Adam the update is lr·m/(√v + ε), and scaling g by 32 scales both m
and √v by 32, so the ratio barely moves — the only trace is ε becoming effectively ε/32. It
shifts the loss by 5e-06. Any tolerance loose enough to survive honest float drift waves it
straight through.

That is what made the bitwise assertions worth the design effort rather than being a nicety. It
is only possible because `all_reduce` is *built* as reduce-scatter plus all-gather — so ZeRO-0
sees the same additions in the same order as ZeRO-1/2/3 — and because Adam is element-wise, so
running it on a shard gives bitwise what running it on the whole tensor gives for those
indices. Two design decisions made for correctness turned out to buy discriminating power.

**The float drift that remains lives somewhere specific.** Against a reference that accumulates
in plain rank order instead of ring order, the loss differs by 5.96e-08 but individual
parameters by up to 2.5e-04. That looked alarming until I looked at *which* parameters: 0.1% of
elements, concentrated in `attn.bqkv` — biases initialised to exactly zero, where Adam's second
moment is near zero, so ε dominates the denominator and tiny reordering differences get
amplified. Not a bug, and not something a single flat parameter tolerance would have described
honestly.

**Small groups shard terribly.** The final LayerNorm is 128 elements. Padded to a multiple of
32 ranks it becomes 512 — **75% of that group is padding**, to split a LayerNorm 32 ways. Across
the whole model the damage is mild, because the blocks dominate: Ψ = 269,821 rounds up to
Ψ_padded = 271,360, so 8,480 elements per rank and 0.57% wasted overall. But the ratio gets
worse as N grows, and a model with many small tensors would be hit far harder. I arrived at the
problem by measuring it and then found that FSDP1 already solves it by flattening a wrapping
unit into one `FlatParameter`, and DeepSpeed by leaving small tensors replicated below
`stage3_param_persistence_threshold`.

**ZeRO-3's transient buffer nearly made my scaling table a lie.** While a layer is gathered,
that layer's full parameters *and* its full gradient are resident, and that term does not shrink
with N. Writing the ZeRO-3 row as 16Ψ/N alone claims that 80 GiB per GPU at N = 2048 trains an
8-trillion-parameter model. With the transient term it is 330 billion — **25× smaller**. On this
model the effect is stark for the opposite reason: ZeRO-3's persistent state is 135,680 bytes
and its transient buffer peaks at 266,240, so the thing that is not sharded is twice the thing
that is. That is a small-Ψ pathology and it mostly disappears at real scale, which is worth
saying rather than presenting as a general finding.

**Two thirds of the runtime was the measurement, not the work.** The activation tracker fires a
Python callback per saved tensor — 112 per forward, 32 ranks — and `weakref.finalize` costs
about 0.3 ms a call. Replacing it with `__del__` and tracking on one rank by default (activation
memory is identical across ranks, which is asserted rather than assumed) took a step from
1,595 ms to 524 ms. The remaining gap between that and the 89 ms the same arithmetic takes
single-threaded is GIL contention, and it is exactly why no timing here is presented as a
result.

## Checking against real PyTorch

A simulator that only agrees with itself proves nothing.
[`scripts/torch_crosscheck.py`](scripts/torch_crosscheck.py) runs the same model under
`torchrun --nproc_per_node=4` on the gloo backend with real `DistributedDataParallel` and real
FSDP2 `fully_shard`. Both sides call the same `vzero.model.forward`, so a mismatch cannot be a
difference in the model.

**It validates numerics only, and that limit is deliberate.** gloo has no native reduce-scatter:
`ProcessGroupGloo` implements `reduce_scatter_single` by cloning the input, running a full
all-reduce, and copying out this rank's chunk. So a gloo reduce-scatter moves 2(N−1)/N bytes
where the algorithm wants (N−1)/N, and real FSDP on gloo moves roughly 4(N−1)/N per step rather
than 3. Instrumenting torch's byte counters and comparing them against the 1.5× result would
look like a refutation of a derivation that is correct. Every communication number in this
project comes from our own fabric, which implements a real ring.

| | world = 1 | world = 4 |
|---|---|---|
| our ZeRO-0 vs torch DDP | 4.768e-07 ✅ | 8.345e-07 ✅ |
| our ZeRO-3 vs torch FSDP2 | 4.768e-07 ✅ | 2.248e-01 ❌ |
| torch DDP vs torch FSDP2 | 0.000e+00 ✅ | 2.248e-01 ❌ |

**Our ZeRO-0 matches PyTorch's own DDP** at both world sizes. That one check validates the
fabric, the model, the data pipeline and the baseline against an independent implementation at
once, and it is the load-bearing result.

FSDP2 is the more interesting row, and the answer was not what I expected. It *runs* on
gloo/CPU — the folklore that FSDP requires NCCL traces to a 2022 issue that has since been
closed, and torch 2.14 has explicit CPU branches (`_get_device_from_mesh` opens with
`if mesh.device_type == "cpu"`, and `torch.cpu.Stream`/`Event` are no-op shims carrying exactly
the methods FSDP2 calls). At world = 1 torch's DDP and FSDP2 agree bitwise and both match this
simulator, so FSDP2's sharding and gathering are fine.

At world = 4 they diverge — and **torch's own DDP and FSDP2 disagree by the same amount**, which
places the gap on torch's side of the comparison rather than mine. The cause is structural:
FSDP2 registers its post-backward gradient reduce-scatter through the wrapped module's forward
*inputs*, and this model's root module takes a single integer token tensor with no gradient, so
the hook never fires and gradients are never reduced across ranks. A conventionally written
`nn.Module` transformer, whose blocks each receive a float activation, does not have this
problem. There is also a related warning FSDP2 emits and which I did have to fix: a wrapped
module returning a *view* tensor silently drops the pre-backward hook, so the output needs a
`.clone()`.

I could make the row green by writing a second, conventional model for the FSDP side. I have
not, because then a mismatch could be the model rather than the parallelism — the one thing
this script exists to rule out. So the claim stays narrow and true:

> ZeRO-0 is validated against PyTorch's DDP. ZeRO-1, ZeRO-2 and ZeRO-3 are validated against a
> single-rank reference and against each other, bitwise. They are **not** validated against
> PyTorch's FSDP.

## What I left out, and where it would go

**ZeRO-Offload** (arXiv:2101.06840) moves optimizer state and the fp32 update to CPU DRAM. It
attaches at the ZeRO-2 boundary above and trades PCIe bandwidth for HBM capacity.

**ZeRO-Infinity** (arXiv:2104.07857) adds NVMe as a third tier plus bandwidth-centric
partitioning, which removes the memory ceiling in the scaling table and replaces it with a
bandwidth one.

**Activation checkpointing** (arXiv:1604.06174) is the orthogonal axis that addresses the red
segment in the memory figure — the one thing ZeRO does nothing about. O(√L) memory for one
extra forward pass; selective recompute in arXiv:2205.05198 picks which tensors to keep.

**Tensor parallelism** (Megatron-LM, arXiv:1909.08053) shards the *compute*, which ZeRO never
does, and therefore shards activations, which ZeRO cannot.

**PyTorch FSDP** (arXiv:2304.11277) is ZeRO-3 with `reshard_after_forward` exposed —
`SHARD_GRAD_OP` is roughly ZeRO-2, `NO_SHARD` is DDP, `HYBRID_SHARD` trades memory for
intra-node locality. FSDP2's move from a single `FlatParameter` to per-parameter DTensor
sharding changes the padding story above.

**DeepSpeed** exposes the knobs this project ran into from first principles:
`stage3_param_persistence_threshold` leaves small tensors replicated (the 75%-padding LayerNorm),
and `stage3_prefetch_bucket_size` sets how many layers are gathered ahead — which raises peak
memory by one layer per unit of depth, the same term that flattens the ZeRO-3 scaling curve.
**ZeRO++** (arXiv:2306.10209) cuts volume roughly 4× with quantized weights and hierarchical
partitioning, which breaks the clean 2Ψ/3Ψ accounting everything here rests on.

## Sources

Every claim attributed to a paper was checked against the paper, not recalled. The ZeRO figures
quoted above — Table 1's 120 / 31.4 / 16.6 / 1.88 GB at Ψ = 7.5B and 2048 / 536 / 284 / 32 GB at
128B, the 2Ψ + 2Ψ + KΨ = 16Ψ formula with K = 12, the 1.5B-on-32GB claim, and the statements
that stages 1 and 2 incur "no additional communication" while stage 3 incurs "a maximum of 1.5x
communication" — were read from arXiv:1910.02054 and are reproduced by
`vzero.analysis.paper_table()`, which the notebook asserts against them.

| Work | arXiv | Used for |
|---|---|---|
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | [1910.02054](https://arxiv.org/abs/1910.02054) | the three stages, Table 1, the communication claims |
| PyTorch FSDP: Experiences on Scaling FSDP | [2304.11277](https://arxiv.org/abs/2304.11277) | `reshard_after_forward`, sharding strategies |
| ZeRO-Offload | [2101.06840](https://arxiv.org/abs/2101.06840) | mentioned only |
| ZeRO-Infinity | [2104.07857](https://arxiv.org/abs/2104.07857) | mentioned only |
| ZeRO++ | [2306.10209](https://arxiv.org/abs/2306.10209) | mentioned only |
| Training Deep Nets with Sublinear Memory Cost | [1604.06174](https://arxiv.org/abs/1604.06174) | activation checkpointing |
| Reducing Activation Recomputation in Large Transformer Models | [2205.05198](https://arxiv.org/abs/2205.05198) | selective recompute |
| Megatron-LM | [1909.08053](https://arxiv.org/abs/1909.08053) | tensor parallelism contrast |
| PyTorch Distributed: Experiences on Accelerating Data Parallel Training | [2006.15704](https://arxiv.org/abs/2006.15704) | DDP gradient bucketing |
| Adam | [1412.6980](https://arxiv.org/abs/1412.6980) | the optimizer, and its scale invariance |

Bandwidth-optimal ring all-reduce has no arXiv entry: Patarasuk & Yuan, *"Bandwidth optimal
all-reduce algorithms for clusters of workstations"*, JPDC 69(2), 2009.

Claims about PyTorch internals were read from the installed source at torch 2.14.0, not from
documentation: `ProcessGroupGloo.cpp` for gloo's reduce-scatter,
`torch/distributed/fsdp/_fully_shard/_fsdp_init.py` and `_fsdp_collectives.py` for FSDP2's CPU
branches and its use of `_unsafe_preserve_version_counter`, and `torch/cpu/__init__.py` for the
no-op `Stream`/`Event` shims.

## Running it

On Colab — open the notebook and run all. The first cell fetches this folder and uses the
preinstalled torch; nothing else is needed, and it takes about two minutes on a CPU runtime.

Locally:

```bash
git clone https://github.com/KarthikeyanMohanraj17/ERA_LLM_training.git
cd ERA_LLM_training/session-12-distributed-training
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest tests/ -q                    # unit tests
.venv/bin/python scripts/make_results.py --steps 20     # figures + results/
OMP_NUM_THREADS=1 .venv/bin/torchrun --nproc_per_node=4 \
    scripts/torch_crosscheck.py                         # real DDP and FSDP2
.venv/bin/jupyter lab zero_32_virtual_gpus.ipynb
```

Python 3.12 is required because torch has no 3.14 wheels. The committed notebook already
carries its outputs, so nothing has to be run to read it.

## Structure

```
zero_32_virtual_gpus.ipynb   the notebook, with outputs from the run that made the figures
README.md                    this file

vzero/
  env.py          set_num_threads(1) and set_num_interop_threads(1) before any worker
                  starts -- 32 workers x 12 intra-op threads oversubscribes a 12-core
                  machine 32x, and is a source of nondeterminism this project asserts
                  against
  vgpu.py         MemArena: counts bytes, keyed by storage address so views are free,
                  holding a strong reference so a recycled address cannot alias.
                  Every memory number in the project depends on this file being right
  accounting.py   activation memory via saved_tensors_hooks. Separate from vgpu.py
                  because it answers a different question -- vgpu counts what we
                  allocate, this counts what autograd decides to keep
  fabric.py       the collectives. all_reduce is built AS reduce_scatter + all_gather
                  rather than as its own algorithm, which is what makes the bitwise
                  claims possible. Two implementations, asserted bit- and
                  byte-identical: a real 31-step ring, and a fast path for the runs
  shard.py        per-group flat layout, alignment, padding. All four modes share one
                  grouping because the ring's addition order depends on which chunk an
                  element lands in
  model.py        a functional transformer -- forward(cfg, ParamAccess, ids). No
                  nn.Module, because with one the parameters always exist and ZeRO-3
                  cannot be demonstrated honestly
  optim.py        Adam on a shard. Element-wise, which is the other half of the
                  bitwise argument
  engine.py       the four modes and the single-rank reference. Two references: one
                  accumulating in ring order (bitwise comparable) and one in plain
                  rank order (independent, and measures the drift)
  analysis.py     closed forms only, no measurement, so the notebook can put the two
                  side by side and assert they agree
  validate.py     the 32 assertions, and the four deliberate bugs
  report.py       figures. Deliberately contains no wall-clock plot
  pool.py         worker threads with abort propagation, so one rank raising fails all
                  32 in milliseconds rather than 32 sequential timeouts

scripts/
  make_results.py      regenerates everything in figures/ and results/
  torch_crosscheck.py  real DDP and FSDP2 under torchrun, numerics only
  build_notebook.py    generates the .ipynb, so notebook code is never a second copy

tests/            unit tests, including the one asserting a divergent rank raises
                  rather than hanging 32 threads
figures/          committed, so the repository page shows outputs without a runtime
results/          metrics.csv, summary.json, crosscheck-world{1,4}.json
```

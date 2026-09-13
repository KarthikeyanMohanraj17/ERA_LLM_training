# ERA_LLM_training

Work from the ERA LLM training programme, one folder per session.

| Folder | Topic |
|---|---|
| [`session8_attention-timeline`](session8_attention-timeline) | A chronological, source-checked tour of attention mechanisms, 2017 → 2026 — 28 mechanisms, each with the problem it solved, the cost it created, and the mechanism that answered that cost. Interactive single-page site, no build step. |
| [`session-12-distributed-training`](session-12-distributed-training) | ZeRO-1, ZeRO-2 and ZeRO-3 built from scratch on 32 virtual GPUs made of CPU threads — a byte-accurate memory ledger, hand-written ring all-reduce / reduce-scatter / all-gather, and measured communication volume checked against the paper's own formulas. All four modes come out bitwise identical to a single-rank reference. Notebook with outputs, plus a cross-check against real PyTorch DDP and FSDP2. |

# Attention Timeline Webapp

A static, single-page chronological tour of attention mechanisms (2017 → 2026), built for a course assignment. Each mechanism gets one `<section class="mech">` with sourced prose and an interactive widget.

## Language

**Mechanism-widget**:
The interactive visual that demonstrates how the mechanism being introduced in a section actually works — the widget most sections already have.
_Avoid_: "the widget", "demo"

**Before-widget**:
A compact, non-interactive recap of the prior mechanism's own widget/state, embedded in a new section so the reader doesn't have to scroll up to see what's being improved on. Only added when the predecessor concept has no widget of its own anywhere on the page. Never a full duplicate rebuild of an existing widget.
_Avoid_: "context widget", "recap widget"

**Cost-widget**:
A visualization of a mechanism's downside. Only built when the cost is a real, quantifiable trade (e.g. quality vs. cache size, resolution lost vs. context gained) — rendered as a small widget or meter. A qualitative, non-numeric cost (e.g. "not peer reviewed," "one more hyperparameter") stays a styled callout, never a decorative chart with no real data behind it.
_Avoid_: "downside viz", "con chart"

**Reused figure**:
A Mechanism-widget or Cost-widget adapted from a diagram that already exists in the mechanism's own primary source (paper, blog post), redrawn in this app's visual style and credited inline ("visualization adapted from Figure N, [paper]"). Preferred over inventing a new metaphor when the source already solved the explanation problem. Never a copy-pasted image — always redrawn/recreated in-house.
_Avoid_: "borrowed widget", "paper figure"

**Topic bullets**:
The Problem / Fixed / Cost / When-to-choose blocks in each section, written as 3-5 one-sentence bullets per topic instead of a single paragraph. Same or greater information density as the prior paragraph form, just decomposed — not simplified content, simplified sentence structure.
_Avoid_: "explanation text", "flow-step prose"

**Chart frame**:
The shared axis layer — `chart()` in `js/helpers.js`. Draws axis lines, ticks, tick labels
and rotated axis titles, and returns `px(v)` / `py(v)` data→pixel mappers. Every widget that
plots numbers goes through it, so no widget invents its own axis conventions. Companions:
`legend()`, `caption()`, `tokenAxis()` (the 1-D position axis for row-of-token widgets),
and `font()` (canvas cannot resolve `var(--mono)` in `ctx.font`, so the family is resolved
once and real font strings built from it).
_Avoid_: "the axis helper", "plot utils"

**Worked example**:
The hand-computable dry run in every section, structured Given → numbered steps → Answer →
note. Small numbers a reader can verify on paper, showing the mechanism as a calculation
rather than a claim. One per section, all 24. Distinct from a Mechanism-widget: the widget
is interactive and explores; the worked example is fixed and proves.
_Avoid_: "the math bit", "example box"

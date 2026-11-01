# P5.11 — stabilized RESCALED oracle consolidation

**Stage stopped at P5.11.1 by a declared stopping rule. P5.11.2, P5.11.3 and
P5.11.4 were not run.** Only completed evidence is reported below.

---

## 1 — Mac Studio repository and host

| | |
|---|---|
| hostname | `Micaels-Mac-Studio.local` (macOS 26.6.2, arm64) |
| repository | `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation` |

The path `/Users/micaelsimoes/PycharmProjects/shared-resources-planning` does not
exist on this machine and was neither inspected nor modified. Machine-specific
paths recorded in the P5.10 report are treated as historical provenance and were
not rewritten.

## 2 — Branch, HEAD, upstream, working tree

| | at stage start | at stage end |
|---|---|---|
| branch | `feature/derivative-free-planning` | `feature/derivative-free-planning` |
| HEAD | `fbdb2f6a7a6a88eab6d1bdaeb569c6e070eaf1dd` | `fbdb2f6a7a6a88eab6d1bdaeb569c6e070eaf1dd` |
| tracked modifications | 0 | **0** |
| upstream | `origin/feature/derivative-free-planning` | unchanged |
| divergence | **ahead 7, behind 0** | unchanged |

No uncommitted user work was present. Nothing was committed, merged, pulled,
pushed, rebased or cherry-picked.

Latest 15 commits:

```
fbdb2f6a P5.10: stabilized rescaled ADMM oracle established
0bfeb1ae Re-restore governing documents after merge 611059b9
611059b9 Merge remote-tracking branch 'origin/feature/derivative-free-planning' ...
518aa1d8 .md files updated.
df77a9b9 Restore governing documents and record P5.9
fe6b5c63 P5.9: rescaling fixes coverage and the landscape, coordination does not transfer
c06f2cd4 Pin IPOPT to the local binary; sweep migration paths; correct licence
1c352bd3 Housekeeping: migration paths, gate hardening, environment manifest
bff465d7 P5.8: ADMM objective scaling validated, but it is necessary and not sufficient
f6db8fce P5.7: branch-selection diagnosis — objective scaling, not branch multiplicity
668de111 P5.6-D9/D14: K=12 escalation and the search-readiness decision
a5171405 P5.6-D4..D12: depth sweep K=2/4/8 plus terminal self-refinement
64b23359 P5.6-D0/D1: uniformly refined oracle H_K defined; record corrections
22c13662 P5.6-C3..C9: landscape unstable; continuation fixes coverage but shifts the branch
f09390a4 P5.6-C1/C2: landscape is not stable across deterministic template generations
```

## 3 — Planning-document history findings

**The Mac Studio documents are not stale.** Inspected independently, not inferred
from any other checkout:

| document | lines | sha256 (16) | P5.6-D | P5.7 | P5.8 | P5.9 | P5.10 | stale P5.6-C heading |
|---|---|---|---|---|---|---|---|---|
| `REVISION_CONTEXT.md` | 1607 | `a492cc9e8e6202cf` | 8 | 17 | 14 | 11 | **0** | no |
| `LOCAL_NLP_STABILITY_PLAN.md` | 1816 | `595fef0da4fdd461` | 6 | 18 | 9 | 5 | **0** | no |

Both were last modified by `0bfeb1ae`. All four referenced commits exist in this
checkout: `df77a9b9`, `0bfeb1ae`, `611059b9`, `518aa1d8`.

`518aa1d8` ("`.md files updated.`", authored 2026-09-09 18:56 +0100, parented
directly on the P5.8 commit `bff465d7`) replaced both documents with a
P5.6-C-era snapshot. It reached this checkout through merge `611059b9` and was
reverted locally by `0bfeb1ae`. **`origin/feature/derivative-free-planning` is
still at `518aa1d8`**, which is why the documents read as obsolete from origin
while being correct here.

**No documentation restoration was prepared, because none is required.** The
correct later versions are already the working versions. What the documents lack
is a P5.10 section — that is new authoring rather than restoration, and it was
not performed under this stage's change-control rules. Documentation was left
byte-for-byte unchanged.

## 4 — P5.10 artifact inventory

All present, git-tracked, unmodified.

| artifact | bytes | sha256 (16) |
|---|---|---|
| `p510_oracle.py` | 13 814 | `6575217fcfcae5b1` |
| `p510_a_state.py` | 9 093 | `cfa0c664db2eaed3` |
| `p510_b_fixedrho.py` | 5 225 | `5f092086f017bbcd` |
| `p510_c_endpoint.py` | 7 718 | `863bbb8b0e0e846a` |
| `p510_e_criteria.py` | 12 576 | `a778846c66ec1e1f` |
| `p510_f_replay.py` | 4 597 | `c3dcf50725ef7126` |
| `p510_g_anchor.py` | 3 597 | `4e3b5d99e0752cbe` |
| `P5_10_..._REPORT.md` | 26 015 | `fe674ee534ba9b7c` |

`data/SRP1/Results/P510/` holds 21 tracked files (12 JSON, 9 filtered logs,
460 KB). Per-file SHA-256 for all of them is recorded in
`data/SRP1/Results/P511/p511_0_recovery.json`. Nothing missing, nothing extra,
nothing modified. The P5.10 report contains no separate hash manifest to compare
against, so the comparison is against the git index.

## 5 — Canonical Mac Studio runtime

Discovered, not assumed. Ten interpreters were enumerated; `/opt/anaconda3` does
not exist on this machine. Exactly one satisfies the accepted provenance:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python
```

| field | value |
|---|---|
| Python | 3.11.11 |
| NumPy / pandas / SciPy / Pyomo | 2.4.2 / 3.0.0 / 1.17.0 / 6.9.5 |
| copulas | 0.14.0 |
| IPOPT | **3.14.18**, ASL **20241111**, at `/usr/local/bin/ipopt` |
| linear solver | **ma97** |
| scenario checksum | `5a02b77c…10358`, matches canonical |
| platform | macOS 26.6.2, arm64 |
| solver-affecting env vars | none set; `NLP_SOLVER_PATH` in `.env` resolves the IPOPT binary |

Nearby environments differ materially and were not used: `opf_env_py312`
(Pyomo 6.10.0, copulas 0.14.1), `srp_env` (documented noncanonical),
`opf_env_py311_pre_migration` (no copulas). Per the instruction to gate "until
one environment exactly satisfies", enumeration stopped at the verified match.

The R0 gate asserts five identities — scenario checksum, IPOPT path, IPOPT
version, ASL build and HSL linear solver — and passed.

## 6 — Frozen reproduction gates

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p511_0_recovery.py
```

| gate | expected | observed | absolute Δ | relative Δ |
|---|---|---|---|---|
| CURRENT polished total | `828021090.3608505` | `828021090.3608505` | **0.0** | **0.0** |
| RESCALED pre-polish recourse | `825814074.4930633` | `825814074.4930633` | **0.0** | **0.0** |

**Both are bit-identical.** No floating-point explanation is involved and no
acceptance threshold was applied or widened. CURRENT terminated `VALID` in 2
cycles; RESCALED terminated `POLISH_FAILURE` in 4 cycles, which is the accepted
P5.8-C generation-1 behaviour and is what that gate measures.

**P5.11.0 passed. No stopping rule fired.**

## 7 — Commands executed after the provenance gate

```
python p511_1_selfconsistent_t0.py build
python p511_1_selfconsistent_t0.py compare "base"
python p511_1_selfconsistent_t0.py compare "se|node5|2025|-10%"
python p511_1_selfconsistent_t0.py compare "se|node9|2025|-10%"
```

The three `compare` invocations ran concurrently, one per candidate, inside the
P5.6-B7 four-worker ceiling. Each process loads its own template copy and its own
deep copy of the planning problem; every solve is deterministic given its inputs.

## 8 — Exact code and documentation diff

**Tracked files changed: none.** Verified individually for the production
sources, `data/SRP1/SRP1_params.json`, all seven P5.10 harnesses, the P5.10
evidence directory and both planning documents — all `unchanged`.

New untracked files only:

```
p511_0_recovery.py                     P5.11.0 recovery, inventory and gates
p511_1_selfconsistent_t0.py            P5.11.1 template build and comparison
data/SRP1/Results/P511/                evidence directory
p511_0.log, p511_1_*.log               console logs
```

Nothing was committed.

## 9 — P5.11.1: self-consistent RESCALED T0

### Construction

Built cold via production's own `run_operational_planning(type='distributed')`
with `p58_rescale.patched_admm_objectives` active, so the 48 augmented objectives
are RESCALED **at construction** rather than retrofitted onto a
CURRENT-formulation template. ρ set to `1.5 / 300 / 1.0` in the deep copy before
construction; adaptive ρ disabled; all four carried-state channels neutralised
plus `candidate_solution`.

| field | value |
|---|---|
| template id | `P511-SELFCONSISTENT-T0` |
| state SHA-256 (consensus + dual) | `14a7ea85ce283a595a35b1768c93c2a60bda872d4f5138c5168b3691a625cbf0` |
| file SHA-256 | `3629784b277724e1f2c406f4a6dbd4599ea92b9d7ee126ac67bb2148909a67bd` |
| config hash | `a080612dedf2727e` |
| blocks rescaled at build | 48 |
| `effective_scale` range | `[9.41457e4, 1.16024e5]` |
| ρ in built template | `{v: 1.5, pf: 300.0, ess: 1.0}` — undecayed, adaptive off |
| artefact | `data/SRP1/Results/P511/p511_selfconsistent_t0.pkl` (172 MB, not committable) |

The `effective_scale` range matches P5.7 and P5.8 exactly, confirming the build
path is the same one those stages measured.

### The construction did not converge

| build | cycles | final `cycle_convergence` |
|---|---|---|
| inherited T0 — cold under **CURRENT** | 17 | **True** |
| diagnostic T0 — cold under **RESCALED** | **25 = `num_max_iters`** | **False** |

The cold RESCALED build exhausted the production iteration cap without
converging, and was still moving steeply when it stopped:

| cycle | `primal_pf` | recourse | converged |
|---|---|---|---|
| 1 | 4.5667e-01 | 2 352 009 862.13 | False |
| 13 | 1.0891e-04 | 1 461 174 062.84 | False |
| 25 | 3.5893e-04 | 1 402 384 338.11 | **False** |

`cycle_convergence` was False at **every one of the 25 cycles**. The frozen
diagnostic template is therefore a non-converged state, and every evaluation
warm-started from it inherits that.

### Comparison — ρ_pf = 300 (principal)

| template | candidate | status | cycles | total objective | polish correction | polish failures |
|---|---|---|---|---|---|---|
| inherited | base | VALID | 6 | 828 008 246.6626 | +73 007.12 | 0 |
| inherited | `se\|node5\|2025\|-10%` | VALID | 6 | 828 007 656.3830 | +73 004.25 | 0 |
| inherited | `se\|node9\|2025\|-10%` | VALID | 6 | 828 007 602.4002 | +73 015.62 | 0 |
| self-consistent | base | VALID | **25** | **1 310 859 830.2811** | −1 583.36 | 0 |
| self-consistent | `se\|node5\|2025\|-10%` | VALID | **25** | **1 310 859 362.3754** | −1 567.62 | 0 |
| self-consistent | `se\|node9\|2025\|-10%` | VALID | **25** | **1 310 859 213.6120** | −5 027.01 | 0 |

Base-relative deltas, and the acceptance test:

| candidate | inherited Δ | self-consistent Δ | \|difference\| | ≤ 22.09 ? |
|---|---|---|---|---|
| `se\|node5\|2025\|-10%` | −590.28 | −467.91 | **122.37** | **FAIL** |
| `se\|node9\|2025\|-10%` | −644.26 | −616.67 | **27.59** | **FAIL** |

### Comparison — ρ_pf = 1000 (isolated cross-check, kept separate)

| template | candidate | status | cycles | total objective | polish correction | polish failures |
|---|---|---|---|---|---|---|
| inherited | base | VALID | 16 | 828 011 656.1647 | +849.83 | 0 |
| inherited | `se\|node5\|2025\|-10%` | VALID | 16 | 828 011 042.5688 | +845.38 | 0 |
| inherited | `se\|node9\|2025\|-10%` | VALID | 16 | 828 011 012.3976 | +843.11 | 0 |
| self-consistent | base | VALID | **25** | **1 371 142 017.9208** | −2 242.90 | 0 |
| self-consistent | `se\|node5\|2025\|-10%` | VALID | **25** | **1 371 141 472.9374** | −8 771.35 | 0 |
| self-consistent | `se\|node9\|2025\|-10%` | VALID | **25** | **1 371 141 455.7047** | −3 944.84 | 0 |

| candidate | inherited Δ | self-consistent Δ | \|difference\| | ≤ 22.09 ? |
|---|---|---|---|---|
| `se\|node5\|2025\|-10%` | −613.60 | −544.98 | **68.61** | **FAIL** |
| `se\|node9\|2025\|-10%` | −643.77 | −562.22 | **81.55** | **FAIL** |

Pre-polish interface disagreement, self-consistent template: `2.227e-04` p.u. at
ρ=300 and `1.597e-04` at ρ=1000, against `9.05e-05` and `1.56e-05` for the
inherited template — worse agreement, consistent with non-convergence. The polish
correction also inverts sign: it *improves* the self-consistent result
(−1 583 to −8 771) where it *costs* on the inherited one (+849 to +73 015),
which is the signature of polishing a state that is not yet at a fixed point.

### Acceptance, applied mechanically

| criterion | result |
|---|---|
| every local and polish block succeeds | **pass** — 0 failures in all 12 runs |
| node-5 / node-9 delta signs unchanged | **pass** — all negative under both templates |
| ordering unchanged | **pass** — `se\|node9` best in all four configurations |
| each Δ difference ≤ 22.09 | **FAIL** — 122.37, 27.59 (ρ=300); 68.61, 81.55 (ρ=1000) |
| rerun independent of call order | not reached |
| no additional untracked state channel | none found |

## 10–14 — Stages not reached

P5.11.2 (expanded coverage), P5.11.3 (interface-stationarity sensitivity) and
P5.11.4 (fixed-ρ certification cross-check) were **not run**. No candidate
manifest was written, no `tau_planning` was updated, no call-order determinism
audit was performed, and no stationarity sensitivity was measured. Reporting any
of them would be fabrication.

## 15 — Failures and stopping-rule decisions

**The P5.11.1 stopping rule that fired:**

> *Stop immediately if … A delta discrepancy exceeds `22.09`.*

It fired for both candidates at both ρ values. The largest discrepancy is
`122.37`, which is `5.5×` the tolerance.

Per the same stage's instruction — *"Do not try a second template construction or
another fix"* — no second construction was attempted, `num_max_iters` was not
raised (a frozen production ADMM setting in any case), and no alternative
numerical route was tried. Execution stopped and the remaining stages were
skipped.

A second, independent stopping condition is also present and worth recording:
the diagnostic template's construction never converged, so it does not meet the
stage's own premise of a "self-consistent" template. The delta test would be the
binding rule regardless.

**Nothing invalidates P5.10.** The inherited-template oracle reproduced both
frozen gates bit-identically in P5.11.0-D, and its P5.11.1 rows are consistent
with P5.10-B to the cent (ρ=300 base `828 008 246.66`; ρ=1000 base
`828 011 656.16`). What failed is the additional hardening experiment, not the
accepted oracle.

## 16 — Recommendation for the next planner action

The finding is specific and, I think, more useful than the experiment that
produced it: **the RESCALED formulation has never been validated on the cold
construction path.** P5.8, P5.9 and P5.10 all ran RESCALED as a warm start from a
template built cold under CURRENT. Building cold under RESCALED does not converge
within the production cap of 25 ADMM cycles — it is still moving by ~5.9e7 in
recourse between cycles 13 and 25.

Three candidate next actions, in the order I would take them:

1. **Diagnose the cold RESCALED convergence path as its own stage** (a P5.11-R or
   P5.12). The question is narrow: does the cold RESCALED ADMM converge at all,
   and at what iteration count and penalty? That is answerable with a bounded
   diagnostic that does not touch production, and it determines whether a
   self-consistent template is achievable in principle. Until it is answered, the
   inherited-template retrofit is the only validated construction.
2. **Decide whether a self-consistent template is actually required.** P5.10
   established that the retrofit is a positive constant multiple, leaving the
   argmin untouched, and that its landscape is stable across depth and penalty.
   The self-consistent template was a hardening step, not a correctness fix. If
   the planner accepts the retrofit on that basis, P5.11.2 could proceed on the
   inherited template with the stage otherwise unchanged.
3. **Push this branch.** `origin/feature/derivative-free-planning` is still at
   `518aa1d8`; this checkout is 7 commits ahead. Everything from P5.9 onward —
   the P5.9 and P5.10 reports, 18 harnesses, all P59/P510 evidence — exists in a
   single unpushed local copy. This is outside my authorization and is flagged,
   not acted on.

I would not begin P5.11.2 on the inherited template without an explicit decision
on (2), because the stage as written predicates it on P5.11.1 passing.

## 17 — Final repository state

```
branch : feature/derivative-free-planning
HEAD   : fbdb2f6a7a6a88eab6d1bdaeb569c6e070eaf1dd
tracked modifications : 0
upstream : origin/feature/derivative-free-planning (ahead 7, behind 0)
```

No production formulation, IPOPT option, ADMM setting, adaptive-ρ rule,
convergence tolerance, objective scaling, proximal regularization, ESSO
degradation, Benders or convex-model file was modified.
`data/SRP1/SRP1_params.json` was never written. No accepted P5.10 evidence was
altered. Nothing was committed, merged, pulled, pushed, rebased or cherry-picked.
No derivative-free search, Benders run or full planning execution was performed.

```
P5.11 PARTIAL — stabilized oracle remains restricted; planner decision required
```

Then stopping for planner review.

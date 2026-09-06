# P5.3-B — Isolated reformulation experiments

Diagnostic only. **No production code was changed and nothing was
productionized.** Each experiment starts from the same accepted production
checkpoint `f77d829359ff…`; no reformulation is stacked on another.

Execution order per the planner's update: **B1 → B2-R → B3**.

| Section | Status |
|---|---|
| **B1 — exact reference-angle gauge** | **Complete** |
| B2-R — RES capability semantics | Pending |
| B3 — active-power ESS prototype | Pending |

---

# B1 — Exact reference-angle gauge

## Provenance

- Script: `p53b1_reference_gauge.py`; raw output
  `data/SRP1/Results/P53B1/p53b1_report.json`.
- Scenario checksum identity unchanged; both branches use the same candidate,
  scenario realization, IPOPT options, MA97, exact Hessian and cold start.
- **A** = production `f_ref ∈ [−1e-5, +1e-5]`; **B** = `f_ref` fixed to exactly
  `0`. Nothing else differs — the hard production `sess_snet_def` equality is
  retained in both, and no P5.2 narrow-band formulation is used.
- `f_ref` is fixed by hooking `Network.run_smopf` and fixing the reference-node
  imaginary voltage immediately before the real solve is delegated to
  production.

**Branch A reproduces the accepted P5 bootstrap baseline exactly** — the same
three persistent failures at `dso/5/2030/Winter`, `dso/5/2035/Winter`,
`dso/9/2025/Summer` — which validates the harness.

## Solver outcomes (51 local solves each)

| | **A (production gauge)** | **B (`f_ref = 0`)** |
|---|---|---|
| DSO succeeded | 33 / 36 | 33 / 36 |
| TSO succeeded | 12 / 12 | 12 / 12 |
| ESSO succeeded | 3 / 3 | 3 / 3 |
| **Persistent failures** | **3** | **3** |
| Primary failures | 3 | **4** |
| Recovery attempts | **0** | **1** |
| Would enter ADMM | No | No |
| Runtime | 276 s | 248 s |
| Iterations — total | 33 073 | **28 922** |
| Iterations — mean | 689.0 | **602.5** |
| Iterations — median | **468.0** | 536.0 |
| Iterations — max | 3000 | 3000 |

## The decisive result: the failures move

The persistent-failure **sets are completely disjoint**:

| Model | **A** | **B** |
|---|---|---|
| `dso/5/2030/Winter` | **FAIL** — maxIterations, 3000 it | **OK** — Optimal, 399 it |
| `dso/5/2035/Winter` | **FAIL** — maxIterations, 3000 it | **OK** — Solved To Acceptable Level, 528 it |
| `dso/9/2025/Summer` | **FAIL** — maxIterations, 3000 it | **OK** — Solved To Acceptable Level, 337 it |
| `dso/5/2025/Spring` | OK — Solved To Acceptable Level, 303 it | **FAIL** — **Error in step computation!**, 544 it |
| `dso/5/2025/Autumn` | OK — Optimal, 409 it | **FAIL** — maxIterations, 3000 it |
| `dso/5/2035/Spring` | OK — Optimal, 742 it | **FAIL** — **Restoration Failed!**, 106 it |

> **`f_ref = 0` fixes all three original P5 bootstrap failures — and breaks three
> previously-successful solves.** The count is unchanged at 3; the identities are
> entirely different.

This is the same *relocation* signature seen with the P5.1/P5.1-B scalar `kappa`
caps, and it is expected: B1 does not touch `sess_snet_def`, which P5.3-A2
established as the sole source of exact equality-rank deficiency.

**A new failure mode also appears.** All three of A's failures are
`maxIterations`. B introduces two modes A never exhibited: **`Error in step
computation!`** and **`Restoration Failed!`**, and requires one recovery where A
required none.

## Derivative diagnostics (corrected P5.3-A2 method)

| Model | Branch | Zero equality rows | σ_min(full) | Reduced σ_min | Reduced condition | `f_ref` column |
|---|---|---|---|---|---|---|
| case33_1/2030/Winter | A | 24 | 0.00 | 5.9246e-03 | 8.983e+04 | norm **1.0** |
| case33_1/2030/Winter | B | 24 | 0.00 | 5.9238e-03 | 8.985e+04 | **absent** |
| case9/2025/Winter | A | 72 | 0.00 | 3.2880e-02 | 1.422e+03 | norm **1.0** |
| case9/2025/Winter | B | 72 | 0.00 | 3.2880e-02 | 1.422e+03 | **absent** |

- **Gauge freedom is removed cleanly.** In B the `f_ref` variable is eliminated
  from the NLP entirely (no column), whereas in A it carries a healthy column
  norm of 1.0.
- **Rank deficiency is unchanged**: 24/72 exactly-zero rows and σ_min(full) = 0
  in both branches, because those rows are all `sess_snet_def`. B1 cannot and
  does not address them.
- **Conditioning is essentially unchanged** — DSO reduced condition rises
  marginally (8.983e+04 → 8.985e+04, +0.02 %); TSO is identical to four
  significant figures.

## Physical / economic comparison — and why it is contaminated

Of 42 comparable objectives, **25 agree to better than 1e-6 relative** (largest
agreeing delta 8.0e-07 relative). The other 17 differ, some very large:
`tso/2030/Winter` +25 970 (5.89× relative), `tso/2025/Autumn` −54 721,
interface `pf_p` up to 66.7 MW and `pf_q` up to 23.4 MVAr, while interface
`vmag` differs by at most 2.4e-04.

**These differences are not attributable to the gauge change.** All 12 TSO
models succeed in both branches, but the TSO cold start is built from the DSO
consensus values, and the DSO *failure set differs between branches*. A failed
DSO solve contributes no interface values, so the TSO models are literally
different problems in A and B. Since initialization completes in neither branch
(`enter_admm = False` both), the downstream economic comparison is contaminated
and **cannot be used to certify "no physical change beyond numerical
tolerance."**

The clean statement available is the one for the 25 models that succeeded in
both branches: they agree to ≤ 8.0e-07 relative, consistent with an exact gauge
restriction of a rotationally-degenerate coordinate.

## Acceptance assessment

| Criterion | Verdict |
|---|---|
| No physical/economic change beyond numerical tolerance | **Not demonstrable** — 25/42 agree to <1e-6, but the remainder is contaminated by the differing failure sets |
| No new failure family appears | **FAILS** — three previously-successful node-5 solves now fail, introducing `Error in step computation!` and `Restoration Failed!`, modes absent from A |
| Gauge freedom removed cleanly | **PASSES** — `f_ref` eliminated from the NLP |
| Initialization robustness / conditioning not worse | **Marginal** — same persistent count (3) but 4 primary failures vs 3 and 1 recovery vs 0; conditioning unchanged (DSO +0.02 %) |

## B1 verdict

**CONTINUE TESTING — do not productionize.**

The gauge fix does exactly what it is supposed to do mathematically (it removes
a genuine ±1e-5 rotational freedom, cleanly, at no conditioning cost) and it
*does* repair all three original P5 failures. But on its own it is not a net
robustness improvement: it relocates the failures onto three different node-5
models and introduces two harsher failure modes. Total iterations fall 12.5 %
while the median rises, so even the effort metric is mixed.

The reason is now well established: B1 leaves the exact zero-gradient
`sess_snet_def` equality untouched, and that is the sole source of exact rank
deficiency. **B1 should be re-evaluated *after* B3**, on top of a formulation
that no longer contains the degenerate rows — where its effect can be isolated
from the dominant defect rather than competing with it.

---

# B2-R — RES capability semantics and conditioning

*Pending.*

# B3 — Active-power ESS structural prototype

*Pending.*

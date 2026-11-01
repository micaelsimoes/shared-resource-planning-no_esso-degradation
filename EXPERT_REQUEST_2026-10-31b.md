# Request for rulings — Addendum 89 (author → expert, 2026-10-31, second request)

## R8 — G-CC-3 slack safety: the bar as worded fails; the mechanism it guards against is safe by 171×

Convention change C1 (salvage term in the ESSO objective) is built and gated (`P5_16_CC_C1C3C4_REPORT.md`,
commit below). G-CC-1 bitwise identity with HEAD held (87/87 blocks, both configurations, both builders); G-CC-2
(new convention value-inert at x = 0; the ESSO NL bitwise identical for C1) held; G-CC-5 (five-block instance
2025–2037, 80 + 3 blocks) held; κ = a/σ = 1/w_med = 4.125e-3 reproduced exactly; the marginal salvage per MWh of
throughput is **0 (2025), −1.23 (2028), −2.36 (2031), −3.59 (2034), −4.85 €/MWh (2037)** — your predictions held.

**The slack-safety check (Addendum 88 R1.4(ii)) failed as worded:** "the salvage marginal per unit of `es_pnet`, in
objective units, ≤ 1e-3 × `PENALTY_ESSO_SLACK`". Per unit of `es_pnet` at one representative period — the unit the
slack pair carries — the marginal peaks at **5.84** objective units (2037 cohort, discharge, Spring; a period there
represents 3 years × ≈ 91 days, so one MW-period is ≈ 1,394 € of salvage); the bar is 1.0; every 2028+ cohort
exceeds it (1.55 / 2.93 / 4.40 / 5.84). Read **per MWh of throughput**, the marginal is 0.005–0.020 objective units,
passing by ≈ 50×. The physical question the check guards — can the ESSO "dispatch" through the P-net slacks more
cheaply than by cycling the cells? — is answered by the ratio penalty/marginal = **171×** (1e3 / 5.84): no, but the
1000× safety factor the wording demands is not met.

Options. **(a)** `PENALTY_ESSO_SLACK` raised to 1e4 for the new instance (ratio 1,712×, margin 1.7 over the bar) —
your own prescribed remedy. Implementation note: the constant lives in `definitions.py`; to keep G-CC-1's switch-off
identity on SRP1 it would become an ESS-params value with today's default (1e3) and 1e4 in the new instance's file
(fix at the writer; one gate re-run). Conditioning: the ESSO objective's scale changes by 10 on the slack entries
only (slacks sit at their bounds at every certified point). **(b)** Rule the bar per MWh of throughput (passes 50×).
**(c)** Re-set the safety factor (171× observed). Planner recommendation: **(a)**, because the bar's unit is the
slack's unit and the fix is a parameter, not a reinterpretation; (b) as the fallback if the expert judges the
MW-period reading wrong.

Two wording confirmations from the same gate: the € parity identity holds in its a-normalised form κ/a = 1/σ (to
3.9e-12 on ESSO, DSO7, TSO) — the literal "1 € moves both equally" differs by a = σ/w_med by design (F1); and
∂objective/∂throughput = −κ·∂S/∂throughput > 0 (the credit is subtracted). Both are recorded as the gate's form.

## Held for Addendum 89, with information
- **R7 (`mu_init` on warm-started network solves):** the A/B shows iterations −78/−79 % and wall −35/−43 % on SRP1
  blocks, but the DSO `mu_init` solves stop 23–41× outside the complementarity bar with 35–38× larger dual
  infeasibility, all within `tol` (a shallower stop, hypothesis). Not proposed until the Part B paper-scale timing
  and the Advisor's read of the T-perf package; the question of what the stopping-slack bar must include goes with it.
- D0: T0.1a–c held every gate (construction exact; Q_cert⁺ correction 13.0 k€ = the recorded detector penalty);
  T0.2's 132 solves are blocked by a Gurobi host-ID mismatch on the 24 GB machine (author action).
- Fixture loadability: 112/139 fixtures unpickle, identical before and after the edit (22 FrozenSMOPF pickles need the
  `s_bounds` symbol deleted in P5.4-B; 5 anchors absent on this machine) — the housekeeping repair of Addendum 88 D-4.

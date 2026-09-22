# P5.15 Addendum 32 Q3, task W30: NOMAD stub check of the OrthoMADS direction and mesh machinery

**Verdict: DONE.** NOMAD 4 installed in a separate environment from a prebuilt Apple-silicon wheel. No source
build was needed. The machinery agrees wherever the two formulations coincide. It differs by design as listed
under "Differences". This checks the direction and mesh machinery on a cheap analytic stub. It is **not** a
check of the SRP1 result. No ADMM or OPF solve was run. The in-house module's
`SolveProfileGuard(permitted=())` is verified at exactly 0 (counts all 0, `verify(0)` failures `[]`, see
`run_console.log`).

Authority: PLANNER_BRIEF Addendum 32 Q3; STEP4_DFO_METHOD.md §6 check (ii) and §5.2/5.3. In-house implementation:
`p515_s47_phase_b_record.py` (last commit f6f86888; sha256 97f573e5...6c62, recorded in `nomad_stub_results.json`).
Repository HEAD at the run: 8f9d67f2.

## Environment (separate; never the canonical one)

- **Separate venv:** `/private/tmp/claude-501/-Users-micaelsimoes-Projects-share-resource-planning-no-esso-degradation/ce1f88a8-5176-446e-89bd-7b0ea6787216/scratchpad/nomad_env`.
  This is session scratch space and is **not committed**.
- **Base interpreter:** Homebrew `/opt/homebrew/bin/python3.12` (Python 3.12.1, macOS-27.0-arm64). It is not conda
  and not `opf_env_py311`.
- **Commands:**
  ```
  /opt/homebrew/bin/python3.12 -m venv <scratchpad>/nomad_env
  <scratchpad>/nomad_env/bin/python -m pip install --upgrade pip
  <scratchpad>/nomad_env/bin/python -m pip install PyNomadBBO    # -> pynomadbbo-4.6.0-cp312-cp312-macosx_11_0_arm64.whl (prebuilt wheel)
  <scratchpad>/nomad_env/bin/python -m pip install pyomo==6.9.5  # the canonical env's pyomo version; the in-house module's guard imports pyomo
  <scratchpad>/nomad_env/bin/python -m pip install numpy
  ```
  The full `pip freeze` is in `separate_env_pip_freeze.txt`: numpy 2.5.3, ply 3.11, PyNomadBBO 4.6.0, pyomo 6.9.5.
  The NOMAD banner reads "Python Interface to NOMAD version 2.2 / NOMAD version 4.6.0".
- **Install route:** the documented pip wheel. It installed in seconds, and no source build was attempted.
- **cp311 wheel:** PyPI also has `pynomadbbo-4.6.0-cp311-cp311-macosx_11_0_arm64.whl`. I only **downloaded** it to
  the scratchpad (`pip download --python-version 3.11 --platform macosx_11_0_arm64 --only-binary=:all:`) and did
  **not** install it anywhere. So STEP4 §6 gate (i), "installs in `opf_env_py311`", has a wheel available, but it
  was **not tested**, because installing into the canonical env is not authorized.
- **Canonical env untouched:**

  | Canonical env measurement | Before | After |
  |---|---|---|
  | `pip freeze` sha256 | `9cb74cbc17200f150c7cb9e801444df6616b438bd7bc3aaa6f5735709880216b` | identical |
  | site-packages listing sha256 | `3f866d07aa073e209c3a6569a7777fc90e7cd978f05e46990094df797194742c` | identical |

  `import PyNomad` in the canonical interpreter still raises `ModuleNotFoundError`.
- **NOMAD 4.6.0 source:** cloned (tag `v.4.6.0`) to the scratchpad for reading only. It is not built and not committed.

## How to run (only from the separate env; the script refuses the canonical env)

```
<scratchpad>/nomad_env/bin/python -u data/SRP1/Results/P515S49/nomad_stub/p515_s49_nomad_stub.py \
    > data/SRP1/Results/P515S49/nomad_stub/run_console.log 2>&1
```

- The run is attached, and both output streams are captured. The wall time is 17 s.
- Each NOMAD run executes in a fresh child interpreter. This works around a PyNomad seeding behaviour; see finding F1.
- The script sets `sys.dont_write_bytecode`, so it writes no cpython-312 `.pyc` into the repository.

## Stub (shared by both implementations)

- **Variables:** the in-house 7-variable lattice z = (zP5, zE5, zP7, zE7, zP9, zE9, zY). Its constraints are the
  in-house `Lattice.reasons`, used verbatim as NOMAD's extreme-barrier (`EB`) output, where the count of violated
  rules must be 0:
  - bounds 0 ≤ z ≤ (10, …, 10, 2);
  - P = 0 ⇔ E = 0;
  - zP ≤ zE ≤ 2 zP;
  - zE ≤ 10;
  - the budget.
- **Stub unit costs:** 400/300 (2025), 320/240 (2030) and 260/190 (2035) EUR per MVA/MWh. These are **not** the
  SRP1 costs. The budget is 1e12, so it is inactive.
- **Stub A (affine):** F = I(z), with positive slopes.
- **Stub B (quadratic):** F = I(z) + 100 Σ_n[(zP_n − tP_n)² + (zE_n − tE_n)²] + 50 (zY − 1)², with tP = (2, 6, 0) and
  tE = (6, 9, 0).
- **Start:** z0 = (4, 8, 4, 8, 4, 8, 0), i.e. 1 MVA / 4 MWh at every node in 2025.
- **Enumerated lattice minima (from `Lattice.domain()`):**
  - A: x = 0, F = 0.
  - B: z = (3, 6, 6, 9, 0, 0, 2), i.e. y2035, n5 0.75/3, n7 1.5/4.5, with F = 2160.
- **In-house:** the production `run_mads`, unchanged. It runs with σ_Q = 0 and bar = 0, so an improvement means a
  strict decrease. Δ0 = 4, Halton t0 = 17, n+1 NEG, and it uses the unit-poll completion with `COMPLETION_CAP` = 30.
  Because that cap stops both stubs, a second in-house arm lifts the cap **in-process only** (to 10^6). That arm is
  labelled `inhouse_cap_lifted`.
- **NOMAD settings:**
  - `DIRECTION_TYPE ORTHO N+1 NEG` and `INITIAL_FRAME_SIZE * 4`;
  - all-integer inputs, so the granularity is 1;
  - `EVAL_OPPORTUNISTIC false`, which gives a full poll;
  - quad-model, Nelder–Mead and speculative search all off;
  - seeds 1, 2 and 3;
  - both `ANISOTROPIC_MESH no` and `yes` (the default).
- **Two kinds of NOMAD run:**
  - **M1:** 12 iterations at `DISPLAY_DEGREE 4`, which exposes the unit-sphere v, the 2n columns, the projected
    directions, the sort and reduction, and the second pass.
  - **M2:** a full run to NOMAD's own termination at `DISPLAY_DEGREE 2`.

## Agreements (M1: 144 polls = 2 stubs × 2 mesh modes × 3 seeds × 12 polls)

| Check | Result |
|---|---|
| M1a Householder | NOMAD's 2n columns equal ±(I − 2vvᵀ) of its printed unit v, in the order H1, −H1, H2, −H2, …. The max abs diff is ≤ 1.60e-6, which is the 6-significant-digit print resolution. This is the in-house formula in `householder_columns`. |
| M1b projection onto the lattice | The in-house rule `round_half_away(Δ·h/‖h‖∞)`, i.e. `project_direction` with mesh 1, reproduces **every** NOMAD "scaled and mesh projected" direction exactly: 144 × 14 directions. It does so from NOMAD's printed columns and from H recomputed from v. For the 57 anisotropic polls this uses NOMAD's per-coordinate Δ_i. NOMAD `GMesh::scaleAndProjectOnMesh` = `roundd(ρ_i·h_i/‖h‖∞)·δ_i`, and `Double::roundd` rounds half away from zero. |
| M1c | Each NOMAD trial point = clip(center + d, bounds), in all 144 polls. |
| M1d the n+1-th direction | NOMAD's rank reduction (greedy, in its printed sort order) is reproduced in 144/144 polls. Its second-pass direction = −Σ(post-snap displacements of the n kept points), **not re-scaled**, reproduced in 73/73 polls where it was generated. Of the 71 polls without a second pass, 70 had a successful first pass (NOMAD then skips the n+1-th direction). The other one had rank 6 < 7 after bound snapping ("Insufficient number of trial points for second pass: 6"). |
| Mesh size | Always 1 in every observed NOMAD poll. The frame sizes seen were 1, 2, 5, 10 and 20. This equals the in-house `MESH_SIZE = 1`. |
| M2 terminal point | Both implementations reach the enumerated lattice minimum on both stubs: all 6 NOMAD runs per stub (3 seeds × 2 mesh modes), and the in-house `inhouse_cap_lifted` arm. On A, NOMAD returns zY = 1 or 2 with zero storage; that is the same point as x = 0, because the year is inactive. |
| Seed consistency | For every M2 run, the first-poll points are among the M1 first-poll points of the same seed. |

## Differences (each with its reason)

- **D1: direction source.** NOMAD 4.6.0 draws v as a normalized standard-normal vector from its seeded RNG
  (`Direction::computeDirOnUnitSphere`). It is **not a Halton point**. The in-house code uses Halton t = 17 + k, as
  in Abramson et al. 2009 and NOMAD 3. So NOMAD 4 **cannot reproduce the in-house directions one for one**. The same
  holds for "the documented OrthoMADS directions" in the Halton sense of STEP4 §6 (ii). What agrees is everything
  downstream of v (M1a–M1c).
  - In-house first poll (t = 17, Δ = 4): (4,0,0,0,0,0,0), (0,1,0,0,0,2,4), (0,0,4,0,0,0,0), (0,0,0,4,0,0,0),
    (0,0,0,0,4,0,0), (0,2,0,0,0,4,−2), (0,4,0,0,1,−2,0), (−3,−4,−3,−3,−3,−3,−2).
  - NOMAD seed-1 first poll (Δ = 5, v = (0.141, −0.624, −0.273, −0.334, −0.040, 0.323, −0.547)): ±(5,1,0,0,0,0,1),
    ±(1,2,−2,−3,0,3,−5), ±(0,−2,5,−1,0,1,−2), ±(1,−3,−1,5,0,1,−2), ±(0,0,0,0,5,0,0), ±(−1,3,1,1,0,5,2),
    ±(1,−5,−2,−3,0,3,3).
  - Every poll's directions are in `nomad_stub_results.json`.
- **D2: composition of the n+1 poll.**
  - NOMAD `ORTHO N+1 NEG` generates the 2n directions ±H. It sorts them (`OrderByDirection`) and keeps the first n
    that raise the rank. The (n+1)-th direction is added only in a **second pass, and only if the first pass
    fails**. It equals −Σ of the kept post-snap displacements and is not re-scaled, so ‖·‖∞ can differ from Δ.
  - In-house always uses the n **+** columns of H, with no ± and no sort or rank selection. Its n+1-th direction is
    the rounded −Σ of the *unrounded* columns, scaled to ‖·‖∞ = Δ, and all n+1 points are evaluated in one full poll.
- **D3: bounds.** NOMAD snaps trial points that fall outside the bounds onto the bounds. It dropped 46 snapped points
  that became equal to the frame centre, and a snap can lower the rank (see M1d). The in-house code rejects such
  points (extreme barrier at zero cost) and does not snap. The duration rules are an extreme barrier in both.
- **D4: poll/frame size ladder.**
  - NOMAD uses the 1-2-5 granular rule: `INITIAL_FRAME_SIZE` 4 becomes **5**. On success it goes 1→2→5→10→20; on
    failure 20→10→5→2→1.
  - In-house (§5.2, ruling A3): Δ0 = 4, doubling and halving through 1↔2↔4↔8.
  - For example, B seed 1 (isotropic): NOMAD frames 5, 10, 5, 2, 5, 2, 5, 2, … ; in-house Δ 4, 2, 4, 2, 4, 2, 1, ….
- **D5: anisotropy.** NOMAD's default (`ANISOTROPIC_MESH yes`) gives per-coordinate frame sizes, e.g.
  (10,10,10,10,5,10,10). The in-house frame is isotropic. `ANISOTROPIC_MESH no` makes NOMAD isotropic.
- **D6: termination.**
  - NOMAD 4.6.0 has **no unit-poll-failure stop for all-granular problems**. At frame 1, a failed poll leaves the
    frame at 1 and draws new random directions. All 12 M2 runs stopped on "Maximum number of total evaluations"
    at MAX_EVAL = 100·3⁷ = 218,700, counting cache hits (the logs show 218,701–218,718).
  - The number of blackbox evaluations to that stop was 356–715 on A and 1,065–1,327 on B. The final best was first
    evaluated at blackbox evaluation 146–522.
  - The in-house code stops on a unit-poll failure that includes the full ℓ∞-1 completion.
- **D7: unit-poll completion (ruling A2) has no NOMAD counterpart.**
  - With the production `COMPLETION_CAP = 30`, the in-house run **stops for review at its first unit poll on both
    stubs**:
    - A: the incumbent is y2035 n5 1/2.5, n7 1.5/4, n9 1.75/3.5, with 863 feasible neighbours.
    - B: the incumbent is y2035 n5 1/4, n7 1.5/4, n9 1/3.5, with 629 feasible neighbours.
  - The budget is inactive in the stub. Under SRP1's €1e6 budget the neighbourhood is smaller; this stub says
    nothing about how often the cap binds there.
  - With the cap lifted, the in-house run reaches the minimum with 2,216 (A) and 1,820 (B) evaluations.
- **D8: canonicalization.** NOMAD does not canonicalize an inactive year. The in-house code does (§5.4).

## Findings for the Planner (observations, not fixed)

- **F1: PyNomad seeding.** With PyNomadBBO 4.6.0, a `SEED` equal to the previous `optimize()` call's `SEED` in the
  same process is not re-applied. That run starts from a fixed default RNG state instead.
  - As a result, in a first trial that alternated M1 and M2 per seed in one process, the M2 runs for seeds 1, 2 and 3
    had identical first-poll points and identical blackbox-evaluation counts (1,245 on B isotropic). Their first poll
    also differed from the same seed's M1 run. The trial outputs were discarded, and the committed outputs come from
    the per-process design.
  - In a scratch test, calling `PyNomad.setSeed(s)` before `optimize()` did not help: it produced the same first-poll
    points for seeds 1 and 2 in every call.
  - The workaround here is one child process per NOMAD run. Any NOMAD-based campaign must do the same, or record the
    RNG state.
- **F2: termination.** STEP4 §3 defines termination as "the poll at unit poll size fails". NOMAD 4.6.0 does not
  implement that for granular variables (D6). Using NOMAD would need its own stop rule, for example `MAX_BB_EVAL`
  or a custom callback.
- **F3: frame ladder.** STEP4 §5.2's "Δ0 = 4, double/halve ... as implemented in NOMAD 4" is internally inconsistent
  with NOMAD 4.6.0: NOMAD rounds 4 to 5 and uses the 1-2-5 ladder (D4).

## Files

- `p515_s49_nomad_stub.py`: the stub check. It refuses to run from the canonical env.
- `run_console.log`: stdout and stderr of the final run.
- `nomad_stub_results.json`: every check per poll, the in-house poll histories, and NOMAD results and parameters.
- `nomad_M1_<stub>_<iso|aniso>_seed<s>.log`: NOMAD `DISPLAY_DEGREE 4` logs for the first 12 iterations. These are
  the primary evidence for M1.
- `nomad_M2_<stub>_<iso|aniso>_seed<s>.log`: NOMAD full-run logs at `DISPLAY_DEGREE 2`.
- `separate_env_pip_freeze.txt`: the separate environment.
- `MANIFEST.sha256`: the sha256 of every file above.

## Time used

The task started at 10:09:56 WEST. The final run finished at about 10:23 WEST. The commit was made at about
11:07 WEST; the first commit attempt failed on a shell word-splitting error and staged nothing. The total is about
57 minutes of wall time, against the 90-minute cap.

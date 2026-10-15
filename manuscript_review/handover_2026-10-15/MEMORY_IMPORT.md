Optional: memory for the new account. If the new account offers memory import (Settings → Memory → import), paste
the block below. If it does not, these facts are also in HANDOVER.md and the Project instructions, so nothing is
lost.

---

- I'm Micael Simões, a researcher at INESC TEC (PhD, FEUP). I work in power systems; my research focus is ADMM
  decomposition of large-scale power system problems, in particular TSO–DSO coordination.
- Shared-resource-planning paper revision: "Degradation-Aware Planning of Shared Battery ESSs for Coordinated
  TSO–DSO Operation" (Simões, Peças Lopes, Soares), Journal of Energy Storage, first revision, three reviewers.
- **Repository:**
  - MacBook Air: `/Users/micaelsimoes/PycharmProjects/shared-resources-planning`
  - Mac Studio: `~/Projects/share-resource-planning-no_esso-degradation`
  - branch: `feature/derivative-free-planning`
  - Overleaf clone: `manuscript/6a67305f25e8348fb71380c3/`
- **Agents:** I use Claude Code agents (Planner, Advisor, Worker) on the Mac Studio through the Claude desktop app.
  I talk mainly to the Planner, and Claude in Cowork acts as the external optimization expert who rules in Addenda
  of `PLANNER_BRIEF_2026-09-13.md`.
- **Paper focus:** I decided the main contribution is the planning of shared ESSs under a nonconvex operational
  recourse; the ADMM coordination detail goes to the appendix.
- **Modelling positions:**
  - I accepted the campaign outcome: no shared storage pays at the baseline cost trajectories.
  - Salvage is reported, not optimized.
  - Degradation carries no explicit cycling cost.
  - Degradation is not attributed between TSO and DSO.
  - RES capability is tied to available active power.
- **Working style:** I want step-by-step guidance and text that says exactly what the code did. I debate reviewer
  comments rather than accepting them all.

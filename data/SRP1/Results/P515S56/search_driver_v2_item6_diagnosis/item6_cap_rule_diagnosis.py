"""Zero-solve diagnosis of the five item 6 failures of the driver gate run search_driver_v2 (from 5d84ccf5).
Reads only the frozen campaign specs the gate wrote under search_driver_v2/item6_settled_key/ and recomputes their eval
keys with the harness's own key functions. Hypothesis: every failure is the cap-in-key rule of 8948db26 (S56 EV-plan
entries' eval keys wrapped LAST as s56_cycle_cap_evaluation_key(., cap)), which the item 6 checks (written at dcea9628,
before that rule) do not account for. Guard armed: SolveProfileGuard(permitted=()), verify(0)."""
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
from p513_solve_profile_guard import SolveProfileGuard
g = SolveProfileGuard(permitted=(), label='item6 cap-rule diagnosis (never solves)').install()
import p515_s44_campaign_harness as H
base = 'data/SRP1/Results/P515S56/search_driver_v2/item6_settled_key'
def spec(prefix):
    d = [x for x in sorted(os.listdir(base)) if x.startswith(prefix)][0]
    p = glob.glob(os.path.join(base, d, 'campaign_spec_*.json'))[0]
    return p, json.load(open(p))
out = {'hypothesis': __doc__, 'settled_key_checks': {}, 'option_off_identity_checks': {}}
for tag, setname in (('ev', 's56_ev_staged'), ('draw0', 's56_draw0_staged')):
    pa, al = spec(f'allowed_{tag}_'); pp, pre = spec(f'pre_{setname}_')
    by = {e['label']: e for e in al['candidates']}; pb = {e['label']: e['eval_key'] for e in pre['candidates']}
    cap = al['cap']; rows = {}
    for n, b in pb.items():
        rows[n] = {'old_check_settled_eq_sk_wrap_of_exit': by[f'{n}_settled']['eval_key'] == H.s56_settled_stop_evaluation_key(by[f'{n}_exit']['eval_key'], True),
                   'exit_eq_cap_wrap_of_pre_base': by[f'{n}_exit']['eval_key'] == H.s56_cycle_cap_evaluation_key(b, cap),
                   'settled_eq_cap_wrap_of_sk_wrap_of_pre_base': by[f'{n}_settled']['eval_key'] == H.s56_cycle_cap_evaluation_key(H.s56_settled_stop_evaluation_key(b, True), cap),
                   'candidate_key_unchanged': by[f'{n}_settled']['key'] == by[f'{n}_exit']['key'],
                   'settled_entry_field_true': by[f'{n}_settled'].get(H.S56_SETTLED_STOP_DISABLED_KEY) is True}
    out['settled_key_checks'][tag] = {'allowed_spec': pa, 'pre_spec': pp, 'cap': cap, 'per_plan': rows,
                                      'absent_no_entry_field': H.S56_SETTLED_STOP_DISABLED_KEY not in by['x0_exit']}
for name in ('s56_ev_staged', 's56_draw0_staged', 'calibration_key_coupled', 'srp1_single_year'):
    pp, pre = spec(f'pre_{name}_'); pq, post = spec(f'post_{name}_')
    fields = sorted({k for a, b in zip(pre['candidates'], post['candidates']) for k in set(a) | set(b) if a.get(k) != b.get(k)})
    top = sorted(k for k in set(pre) | set(post) if k != 'candidates' and pre.get(k) != post.get(k))
    wrap = [b['eval_key'] == H.s56_cycle_cap_evaluation_key(a['eval_key'], post['cap']) for a, b in zip(pre['candidates'], post['candidates'])]
    same_keys = [a['eval_key'] == b['eval_key'] for a, b in zip(pre['candidates'], post['candidates'])]
    out['option_off_identity_checks'][name] = {'pre_spec': pp, 'post_spec': pq, 'entry_fields_differing': fields,
                                               'top_level_fields_differing': top, 'post_eq_cap_wrap_of_pre': wrap,
                                               'post_eq_pre': same_keys}
out['guard_verify_0'] = g.verify(0)
print(json.dumps(out, indent=1, sort_keys=True))

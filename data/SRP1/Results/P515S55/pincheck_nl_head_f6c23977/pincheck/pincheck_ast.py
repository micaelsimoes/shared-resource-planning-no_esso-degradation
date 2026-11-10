"""Pin check f6c23977: AST-level statement of what the switch-off path of each changed production function is,
compared with the same function at 6f43479e. Read-only; imports no project module."""
import ast, json, subprocess, sys, hashlib
REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'


def src(rev, f):
    if rev == 'HEAD-worktree':
        return open(f'{REPO}/{f}').read()
    return subprocess.run(['git', 'show', f'{rev}:{f}'], capture_output=True, text=True, cwd=REPO, check=True).stdout


def find(tree, name, cls=None):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and cls and node.name == cls:
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and n.name == name:
                    return n
        if not cls and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def strip_doc(fn):
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], 'value', None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:]
    return fn


def dump(n):
    return ast.dump(n, include_attributes=False)


class Off(ast.NodeTransformer):
    """Replace each switch-guarded If by its switch-off branch."""
    def visit_If(self, node):
        self.generic_visit(node)
        t = ast.unparse(node.test)
        if t == "getattr(shared_ess_data.params.salvage_value, 'in_objective', False)":
            return node.orelse          # off branch
        if t in ("hasattr(models[node_id], 'salvage_objective_kappa')", "hasattr(model, 'salvage_objective_kappa')"):
            return []                   # off: not entered (attribute never constructed when off)
        if t == 'self.params.available_energy_soh_point is not None':
            return []                   # off: key not declared -> None
        if t == "params_data.get('available_energy_soh_point') is not None":
            return []
        return node


res = {}
old_t = {f: ast.parse(src('6f43479e', f)) for f in ('shared_resources_planning.py', 'shared_energy_storage_data.py',
                                                     'shared_energy_storage_parameters.py')}
new_t = {f: ast.parse(src('HEAD-worktree', f)) for f in old_t}
cases = [('shared_energy_storage_data.py', '_build_subproblem', None),
         ('shared_energy_storage_data.py', '_get_esso_complementarity_diagnostics', None),
         ('shared_energy_storage_data.py', 'read_parameters_from_file', 'SharedEnergyStorageData'),
         ('shared_resources_planning.py', 'update_shared_energy_storage_model_to_admm', None),
         ('shared_resources_planning.py', '_get_operational_recourse_components', None),
         ('shared_energy_storage_parameters.py', '_read_parameters_from_file', None),
         ('shared_energy_storage_parameters.py', 'read_parameters', 'SalvageValueParameters')]
for f, name, cls in cases:
    o = strip_doc(find(old_t[f], name, cls))
    n = strip_doc(Off().visit(find(new_t[f], name, cls)))
    ast.fix_missing_locations(n)
    eq_full = dump(o) == dump(n)
    # signature-insensitive body comparison (update_... gained objective_scale=None)
    eq_body = [dump(s) for s in o.body] == [dump(s) for s in n.body]
    res[f'{f}:{cls + "." if cls else ""}{name}'] = {'off_path_ast_equal_incl_signature': eq_full,
                                                    'off_path_body_ast_equal': eq_body,
                                                    'args_old': [a.arg for a in o.args.args] + [a.arg for a in o.args.kwonlyargs],
                                                    'args_new': [a.arg for a in n.args.args] + [a.arg for a in n.args.kwonlyargs]}
# the one call-site change in _run_operational_planning
o = find(old_t['shared_resources_planning.py'], '_run_operational_planning')
n = find(new_t['shared_resources_planning.py'], '_run_operational_planning')
calls = lambda fn: [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, 'id', None) == 'update_shared_energy_storage_model_to_admm']
res['_run_operational_planning:calls_to_update_esso'] = {'old': [ast.unparse(c) for c in calls(o)], 'new': [ast.unparse(c) for c in calls(n)]}


class DropKw(ast.NodeTransformer):
    def visit_Call(self, c):
        self.generic_visit(c)
        if getattr(c.func, 'id', None) == 'update_shared_energy_storage_model_to_admm':
            c.keywords = [k for k in c.keywords if k.arg != 'objective_scale']
        return c


res['_run_operational_planning:ast_equal_after_dropping_objective_scale_kwarg'] = dump(strip_doc(o)) == dump(strip_doc(DropKw().visit(n)))
# SharedEnergyStorageParameters.__init__ / SalvageValueParameters.__init__: only new attribute assignments
for cls in ('SalvageValueParameters', 'SharedEnergyStorageParameters'):
    o = find(old_t['shared_energy_storage_parameters.py'], '__init__', cls)
    n = find(new_t['shared_energy_storage_parameters.py'], '__init__', cls)
    ns = [dump(s) for s in n.body]
    added = [ast.unparse(s) for s in n.body if dump(s) not in [dump(x) for x in o.body]]
    res[f'shared_energy_storage_parameters.py:{cls}.__init__'] = {
        'old_statements_preserved_in_order': [dump(s) for s in o.body] == [s for s in ns if s in [dump(x) for x in o.body]],
        'added_statements': added}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(res, indent=1))

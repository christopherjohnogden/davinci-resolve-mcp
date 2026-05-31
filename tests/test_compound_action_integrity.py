"""Static integrity checks for compound action dispatch."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = PROJECT_ROOT / "src" / "server.py"


def _action_in_set(test):
    if not isinstance(test, ast.Compare):
        return None
    if ast.unparse(test.left) != "action":
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.In):
        return None
    comp = test.comparators[0]
    if not isinstance(comp, (ast.Set, ast.List, ast.Tuple)):
        return None
    values = set()
    for elt in comp.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            values.add(elt.value)
    return values


def _action_eq_value(test):
    if not isinstance(test, ast.Compare):
        return None
    if ast.unparse(test.left) != "action":
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    comp = test.comparators[0]
    if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
        return comp.value
    return None


def _walk_action_guards(node, active_guards, violations):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and active_guards:
        return
    if isinstance(node, ast.If):
        guard = _action_in_set(node.test)
        action = _action_eq_value(node.test)
        if action is not None:
            missing = [values for values in active_guards if action not in values]
            if missing:
                violations.append((node.lineno, action, missing))
        body_guards = active_guards + ([guard] if guard is not None else [])
        for child in node.body:
            _walk_action_guards(child, body_guards, violations)
        for child in node.orelse:
            _walk_action_guards(child, active_guards, violations)
        return
    for child in ast.iter_child_nodes(node):
        _walk_action_guards(child, active_guards, violations)


def _is_mcp_tool(node):
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", None) == "tool":
            return True
    return False


def test_no_action_branch_is_hidden_by_enclosing_action_set_in_compound_tools():
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
    violations = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_mcp_tool(node):
            _walk_action_guards(node, [], violations)
    assert violations == []

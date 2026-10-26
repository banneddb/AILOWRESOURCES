"""
The hallucination check. Finds every call in generated code and sorts it into:
  api      - the name exists in the corpus's API index
  local    - defined in the generated code itself (the student's own functions)
  ignored  - Python built-ins, standard library, methods of built-in types
  unknown  - none of the above: a likely hallucinated API call
Deterministic, no LLM involved, so the same code always gets the same answer.

Checkers are per language. Only Python exists so far.
"""

import ast
import builtins
import io
import keyword
import re
import sys
import unittest
from dataclasses import asdict, dataclass, field

from .api_index import ApiIndex


def _public(obj) -> set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


PY_BUILTINS = set(dir(builtins))
PY_TYPE_METHODS = set().union(*(_public(t) for t in (
    list, dict, str, set, frozenset, tuple, int, float, complex, bytes, object,
    io.TextIOWrapper, unittest.TestCase, re.Match, re.Pattern,
)))
STDLIB = set(sys.stdlib_module_names)

CODE_BLOCK = re.compile(r"```([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str, language: str = "python") -> list[str]:
    """Pull the code out of ```fenced``` blocks in a Markdown reply."""
    blocks = CODE_BLOCK.findall(text)
    tags = {"python": ("python", "py", "python3")}.get(language, (language,))
    tagged = [body for tag, body in blocks if tag.lower() in tags]
    return tagged or [body for tag, body in blocks if not tag]  # fall back to untagged blocks


@dataclass
class CheckResult:
    apis_used: list = field(default_factory=list)
    hallucinated: list = field(default_factory=list)
    suggestions: dict = field(default_factory=dict)   # hallucinated name -> similar real names
    parse_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _PythonNames(ast.NodeVisitor):
    """Collects what one Python file defines, imports, and calls."""

    def __init__(self):
        self.defined = set()          # functions, classes, methods, variables, parameters
        self.stdlib_modules = set()   # local names bound to stdlib modules: "import math as m" -> m
        self.stdlib_names = set()     # "from math import sqrt" -> sqrt
        self.calls = []               # (name, base) - base is the object name for obj.name(), else None

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        self.defined.update(a.arg for a in node.args.args + node.args.kwonlyargs)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] in STDLIB:
                self.stdlib_modules.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        if node.module and node.module.split(".")[0] in STDLIB:
            self.stdlib_names.update(a.asname or a.name for a in node.names)
        else:
            # "from media import makeSound": the imported name must be a real API name
            self.calls.extend((a.name, None) for a in node.names if a.name != "*")

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.append((func.id, None))
        elif isinstance(func, ast.Attribute):
            base = func.value.id if isinstance(func.value, ast.Name) else None
            self.calls.append((func.attr, base or ""))
        self.generic_visit(node)


def _regex_names(code: str) -> _PythonNames:
    """Fallback for code that doesn't parse as Python 3 (e.g. Python 2 print statements)."""
    names = _PythonNames()
    names.defined.update(re.findall(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE))
    names.calls = [(n, None) for n in re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
                   if not keyword.iskeyword(n)]
    return names


def _classify(name: str, base: str | None, info: _PythonNames, index: ApiIndex) -> str:
    if name.startswith("_"):
        return "ignored"                     # __init__, private helpers
    if base is None:                         # plain call: name(...)
        if name in info.defined:
            return "local"
        if name in info.stdlib_names or name in PY_BUILTINS:
            return "ignored"                 # range(), len(), ... even if a corpus file reuses the name
        return "api" if name in index else "unknown"
    # method call: obj.name(...)
    if base in info.stdlib_modules:
        return "ignored"                     # math.sqrt(...)
    if name in PY_TYPE_METHODS:
        # .append(), .join(), .get(), ... are far more likely list/str/dict methods
        # than API calls, even when some corpus file defines a function with that name.
        return "ignored"
    if name in index:
        return "api"
    return "local" if name in info.defined else "unknown"


def check_python(blocks: list[str], index: ApiIndex) -> CheckResult:
    """`blocks` in order, e.g. [solution, tests]."""
    result = CheckResult()
    parsed = []
    for i, code in enumerate(blocks):
        try:
            info = _PythonNames()
            info.visit(ast.parse(code))
        except SyntaxError as e:
            result.parse_errors.append(f"block {i}: {e.msg} (line {e.lineno}); used the regex fallback")
            info = _regex_names(code)
        parsed.append(info)

    # Definitions only count forward: the tests (a later block) may call the
    # solution's functions, but a fake defined in the tests must not make the
    # solution's call to a nonexistent library function look "local".
    visible = _PythonNames()
    used, unknown = set(), set()
    for info in parsed:
        visible.defined |= info.defined
        visible.stdlib_modules |= info.stdlib_modules
        visible.stdlib_names |= info.stdlib_names
        for name, base in info.calls:
            kind = _classify(name, base, visible, index)
            if kind == "api":
                used.add(name)
            elif kind == "unknown":
                unknown.add(name)

    result.apis_used = sorted(used)
    result.hallucinated = sorted(unknown)
    result.suggestions = {n: s for n in result.hallucinated if (s := index.suggest(n))}
    return result


CHECKERS = {
    "python": check_python,
    # TODO(java): BRIDGES/Less-Java Java code - parse method calls (e.g. with the javalang package).
    # TODO(less-java): Less-Java programs - calls look like Python's, so a regex pass may be enough.
}


def check_code(code: str | list[str], index: ApiIndex, language: str = "python") -> CheckResult:
    if language not in CHECKERS:
        raise NotImplementedError(f"No API check for {language!r} yet")
    blocks = [code] if isinstance(code, str) else code
    return CHECKERS[language](blocks, index)

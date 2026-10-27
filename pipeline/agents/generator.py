"""
Agent 3 (Generate): write the assignment - a description for students, a
reference solution, and unit tests - from the retrieved documentation.

The same agent also does repairs: it gets its earlier files plus what went wrong
(the test output and any functions that don't exist in the library) and returns
corrected files.

Every mode uses this agent. What changes between modes is only what it's given:
no docs (nc), the docs from one search (srag), or the docs and concept plan from
the retrieval loop (agentic).
"""

import json
import re
from dataclasses import asdict, dataclass

from ..api_check import CheckResult
from ..config import PART_ORDERS
from ..llm import call_claude_json
from ..prompts import load_prompt, render
from ..retrieve import Chunk, format_context
from ..runners import ExecResult

PARTS = ("description", "solution", "tests")


def _schema(parts) -> dict:
    # Claude writes the JSON fields in the schema's order, and each part is
    # conditioned on the ones already written. That is what the `order` switch changes.
    return {
        "type": "object",
        "properties": {p: {"type": "string"} for p in parts},
        "required": list(parts),
        "additionalProperties": False,
    }


@dataclass
class Assignment:
    description: str
    solution: str
    tests: str

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_fence(code: str) -> str:
    """Remove a ```python ... ``` fence if Claude wrapped the code in one."""
    match = re.fullmatch(r"\s*```[\w+-]*\n(.*?)\n?```\s*", code, re.DOTALL)
    return match.group(1) if match else code


def _validate(data) -> str | None:
    empty = [p for p in PARTS if not data[p].strip()]
    return f"these parts are empty: {empty}" if empty else None


def _system(cfg, log, *, prompt: str, corpus: str, context: list[Chunk] | None, runner) -> str:
    template, version = load_prompt(prompt)
    log.add_prompt(prompt, version)
    runtime = cfg.library_runtime.get(corpus, "fake")
    setup, setup_version = load_prompt(f"library_{runtime}")
    log.add_prompt(f"library_{runtime}", setup_version)
    setup = render(setup, corpus=corpus, python_version=runner.version,
                   solution_file=runner.solution_file, test_file=runner.test_file)

    # context=None means "no documentation at all" (nc mode).
    docs_prompt = "docs_none" if context is None else "docs_retrieved"
    docs, docs_version = load_prompt(docs_prompt)
    log.add_prompt(docs_prompt, docs_version)
    if context is not None:
        docs = render(docs, context=format_context(context))

    # The docs go in last, so placeholder-like text inside them is never replaced.
    return render(template, corpus=corpus, library_setup=setup, solution_file=runner.solution_file,
                  test_file=runner.test_file, parts_order=", then ".join(PART_ORDERS[cfg.order]),
                  docs=docs)


def plan_text(concepts: list[dict], verdicts: dict) -> str:
    """The agentic concept plan, as the generator sees it."""
    lines = []
    for c in concepts:
        v = verdicts.get(c["name"])
        if not c["api_need"]:
            lines.append(f"- {c['name']}: plain Python")
        elif v and v["verdict"] == "unsupported":
            lines.append(f"- {c['name']}: not supported by the library; leave it out")
        else:
            apis = ", ".join(v["apis_verified"]) if v else ""
            lines.append(f"- {c['name']}: needs {c['api_need']}" + (f" (library functions found: {apis})" if apis else ""))
    return "\n".join(lines)


def _parse(data) -> Assignment:
    return Assignment(**{p: _strip_fence(data[p]) if p != "description" else data[p].strip() for p in PARTS})


def generate(cfg, log, *, corpus: str, request: str, context: list[Chunk] | None, runner,
             plan: str | None = None) -> Assignment:
    system = _system(cfg, log, prompt="generator", corpus=corpus, context=context, runner=runner)
    user = request + (f"\n\nConcept plan:\n{plan}" if plan else "")
    data = call_claude_json(cfg, log, role="generate", system=system, user=user,
                            schema=_schema(PART_ORDERS[cfg.order]), validate=_validate)
    return _parse(data)


def problems_text(execution: ExecResult, check: CheckResult, corpus: str) -> str:
    problems = []
    if not execution.passed:
        problems.append(f"The tests {execution.summary()}. Test output:\n{execution.stderr or execution.stdout}")
    if check.hallucinated:
        names = []
        for name in check.hallucinated:
            similar = check.suggestions.get(name)
            names.append(f"{name}" + (f" (similar real names: {', '.join(similar)})" if similar else ""))
        problems.append(f"These functions are called but don't exist in {corpus}: " + "; ".join(names))
    return "\n\n".join(problems)


def repair(cfg, log, *, corpus: str, request: str, assignment: Assignment, problems: str,
           context: list[Chunk] | None, runner) -> Assignment:
    system = _system(cfg, log, prompt="repair", corpus=corpus, context=context, runner=runner)
    user = (f"{request}\n\nCurrent assignment:\n{json.dumps(assignment.to_dict(), indent=2)}"
            f"\n\nProblems found:\n{problems}")
    data = call_claude_json(cfg, log, role="repair", system=system, user=user,
                            schema=_schema(PART_ORDERS[cfg.order]), validate=_validate)
    return _parse(data)

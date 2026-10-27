"""
The last step of every mode: generate the assignment, then check it.

  Agent 3 (generate) -> hallucination check + run the tests in a sandbox
    tests fail or a function doesn't exist -> Agent 3 repairs, then check + run again
    at most max_repairs times (agentic: max_repair_iters; nc and srag: 0, one try)

Every mode goes through this same code, so a difference in the results comes
from what the generator was given, not from how the result was checked.
"""

from .agents.generator import Assignment, generate, problems_text, repair
from .api_check import check_code
from .api_index import ApiIndex
from .config import Config
from .logger import RunLog
from .retrieve import Chunk, apply_budget
from .runners import get_runner


def generate_and_test(cfg: Config, log: RunLog, *, corpus: str, request: str, chunks: list[Chunk] | None,
                      index: ApiIndex, max_repairs: int, plan: str | None = None) -> Assignment:
    """`chunks=None` means the generator gets no documentation at all (nc mode)."""
    runner = get_runner(cfg)
    context = None
    if chunks is not None:
        context, log.record["context"] = apply_budget(chunks, cfg.context_budget_lines)

    assignment = generate(cfg, log, corpus=corpus, request=request, context=context, runner=runner, plan=plan)

    for attempt in range(max_repairs + 1):   # first try + up to max_repairs repairs
        check = check_code([assignment.solution, assignment.tests], index, cfg.language)
        execution = runner.run(assignment.solution, assignment.tests)
        log.add_attempt(attempt, assignment, check, execution)

        if execution.passed and not check.hallucinated:
            log.record["outcome"] = "passed"
            break
        if attempt == max_repairs:
            log.record["outcome"] = "failed"   # out of repairs: keep the last version
            break
        assignment = repair(cfg, log, corpus=corpus, request=request, assignment=assignment,
                            problems=problems_text(execution, check, corpus), context=context, runner=runner)
        log.record["repair_iterations"] += 1
    return assignment

"""
The agentic pipeline, in two loops.

1. Retrieval loop:
     decompose -> Agent 1 (write queries, search) -> Agent 2 (critique)
       critique says "rewrite_queries" -> Agent 1 again, for the concepts that didn't fit
       critique says "redecompose"     -> make a new concept plan, then search again
       at most max_critique_loops times

2. Generation loop:
     Agent 3 (generate) -> hallucination check + run the tests in a sandbox
       tests fail or a function doesn't exist -> Agent 3 repairs, then check + run again
       at most max_repair_iters times
"""

from dataclasses import dataclass

from .agents.critic import critique
from .agents.decompose import decompose
from .agents.generator import Assignment, generate, problems_text, repair
from .agents.retriever import run_queries, write_queries
from .api_check import check_code
from .api_index import ApiIndex, build_api_index
from .config import Config
from .embeddings import make_embedder
from .logger import RunLog
from .prompts import build_request
from .retrieve import Chunk
from .runners import get_runner


@dataclass
class RetrievalResult:
    concepts: list[dict]
    chunks_by_concept: dict[str, list[Chunk]]
    verdicts: dict[str, dict]
    stop: str   # accepted | cap_reached | no_api_concepts

    def context_chunks(self) -> list[Chunk]:
        """Chunks for the generator: every concept except unsupported ones, without duplicates."""
        seen, chunks = set(), []
        for c in self.concepts:
            if self.verdicts.get(c["name"], {}).get("verdict") == "unsupported":
                continue
            for chunk in self.chunks_by_concept.get(c["name"], []):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    chunks.append(chunk)
        return chunks


def _redecompose_feedback(review: dict) -> str:
    lines = [review["reason"]]
    lines += [f"- {e['name']}: {e['verdict']}. {e['reason']}" for e in review["concepts"] if e["verdict"] != "fits"]
    return "\n".join(lines)


def retrieval_loop(cfg: Config, log: RunLog, *, corpus: str, request: str, index: ApiIndex) -> RetrievalResult:
    embedder = make_embedder(cfg)
    concepts = decompose(cfg, log, corpus=corpus, request=request)
    new_plan = concepts                                   # logged with the iteration that first uses it
    pending = [c for c in concepts if c["api_need"]]      # plain-Python concepts skip retrieval
    attempts: dict[str, list] = {}                        # concept -> earlier queries + critic notes
    chunks_by_concept: dict[str, list[Chunk]] = {}
    verdicts: dict[str, dict] = {}
    stop = "no_api_concepts"

    for iteration in range(cfg.max_critique_loops + 1):   # first try + up to max_critique_loops retries
        if not pending:
            stop = "no_api_concepts"
            break

        queries = write_queries(cfg, log, corpus=corpus, concepts=pending, attempts=attempts)
        found, retrieval_log = run_queries(cfg, corpus, queries, embedder)
        chunks_by_concept.update(found)

        review = critique(cfg, log, corpus=corpus, concepts=pending, results=found, index=index)
        verdicts.update({e["name"]: e for e in review["concepts"]})
        log.add_iteration({"iteration": iteration, "concepts": new_plan,
                           "retrievals": retrieval_log, "critique": review})
        new_plan = None

        if review["action"] == "accept":
            stop = "accepted"
            break
        if iteration == cfg.max_critique_loops:
            stop = "cap_reached"   # out of retries: go on with the best results so far
            break

        if review["action"] == "redecompose":
            concepts = decompose(cfg, log, corpus=corpus, request=request,
                                 feedback=_redecompose_feedback(review))
            new_plan = concepts
            pending = [c for c in concepts if c["api_need"]]
            attempts, chunks_by_concept, verdicts = {}, {}, {}
        else:  # rewrite_queries: search again only for the concepts that didn't fit
            pending = [c for c in pending if verdicts[c["name"]]["verdict"] == "retry"]
            for c in pending:
                v = verdicts[c["name"]]
                attempts.setdefault(c["name"], []).append({
                    "queries": queries.get(c["name"], []),
                    "critic_reason": v["reason"],
                    "critic_suggestion": v["suggestion"],
                })

    log.record["decomposed_concepts"] = concepts
    log.record["retrieval_stop"] = stop
    return RetrievalResult(concepts, chunks_by_concept, verdicts, stop)


def generation_loop(cfg: Config, log: RunLog, *, corpus: str, request: str, retrieval: RetrievalResult,
                    index: ApiIndex) -> Assignment:
    runner = get_runner(cfg)
    context = retrieval.context_chunks()
    assignment = generate(cfg, log, corpus=corpus, request=request, concepts=retrieval.concepts,
                          verdicts=retrieval.verdicts, context=context, runner=runner)

    for attempt in range(cfg.max_repair_iters + 1):   # first try + up to max_repair_iters repairs
        check = check_code([assignment.solution, assignment.tests], index, cfg.language)
        execution = runner.run(assignment.solution, assignment.tests)
        log.add_attempt(attempt, assignment, check, execution)

        if execution.passed and not check.hallucinated:
            log.record["outcome"] = "passed"
            break
        if attempt == cfg.max_repair_iters:
            log.record["outcome"] = "repair_cap_reached"   # out of repairs: keep the last version
            break
        assignment = repair(cfg, log, corpus=corpus, request=request, assignment=assignment,
                            problems=problems_text(execution, check, corpus), context=context, runner=runner)
        log.record["repair_iterations"] += 1
    return assignment


def run_agentic(cfg: Config, *, corpus: str, concepts: str,
                dataset_description: str = "") -> tuple[RunLog, RetrievalResult | None]:
    log = RunLog(cfg, mode="agentic", corpus=corpus, concepts=concepts, dataset_description=dataset_description)
    result = None
    try:
        index = build_api_index(cfg, corpus)
        log.record["api_index_version"] = index.version
        request = build_request(concepts, dataset_description)
        result = retrieval_loop(cfg, log, corpus=corpus, request=request, index=index)
        generation_loop(cfg, log, corpus=corpus, request=request, retrieval=result, index=index)
        log.write(status="ok")
    except Exception as e:
        log.write(status="error", error=f"{type(e).__name__}: {e}")
        raise
    return log, result

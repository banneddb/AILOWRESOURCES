"""
The agentic retrieval loop:

  decompose -> Agent 1 (write queries, search) -> Agent 2 (critique)
    critique says "rewrite_queries" -> Agent 1 again, for the concepts that didn't fit
    critique says "redecompose"     -> make a new concept plan, then search again
    at most max_critique_loops times

Its result then goes to the generation loop (generation.py) with repairs allowed.
"""

from dataclasses import dataclass

from .agents.critic import critique
from .agents.decompose import decompose
from .agents.generator import plan_text
from .agents.retriever import run_queries, write_queries
from .api_index import ApiIndex
from .config import Config
from .embeddings import make_embedder
from .logger import RunLog
from .retrieve import Chunk


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

    def plan(self) -> str:
        return plan_text(self.concepts, self.verdicts)


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

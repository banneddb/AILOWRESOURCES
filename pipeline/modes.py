"""
The experiment conditions. Every mode starts from the same request and ends in
the same generate -> check -> test step (generation.py), so their logs can be
compared directly. The only differences are the ones being tested:

  mode      docs given to the generator               concept plan   repairs
  nc        none (just the library's name)             no             0
  srag      top_k chunks from one search (the request) no             0
  agentic   chunks from the retrieve/critique loop     yes            max_repair_iters

`order` and `context_budget_lines` apply to every mode the same way.
TODO(rag_da): RAG over LLM-written API descriptions stored in the vector DB.
"""

from .agentic import RetrievalResult, retrieval_loop
from .api_index import build_api_index
from .config import Config
from .embeddings import make_embedder
from .generation import generate_and_test
from .llm import RefusedError
from .logger import RunLog
from .prompts import build_request
from .retrieve import retrieve


def run(cfg: Config, *, corpus: str, concepts: str,
        dataset_description: str = "") -> tuple[RunLog, RetrievalResult | None]:
    """Run one assignment generation in cfg.mode and log it. Returns (log, agentic retrieval result or None)."""
    log = RunLog(cfg, mode=cfg.mode, corpus=corpus, concepts=concepts, dataset_description=dataset_description)
    retrieval = None
    try:
        # The API index is only used to check the generated code; it's never shown to Claude.
        index = build_api_index(cfg, corpus)
        log.record["api_index_version"] = index.version
        request = build_request(concepts, dataset_description)

        if cfg.mode == "nc":
            chunks, plan, max_repairs = None, None, 0

        elif cfg.mode == "srag":
            # One search: the whole request is the query (no decomposition).
            chunks = retrieve(cfg, corpus, request, make_embedder(cfg))
            log.add_iteration({
                "iteration": 0, "concepts": None, "critique": None,
                "retrievals": [{"concept": None, "query": request, "chunks": [c.log_entry() for c in chunks]}],
            })
            plan, max_repairs = None, 0

        else:  # agentic
            retrieval = retrieval_loop(cfg, log, corpus=corpus, request=request, index=index)
            chunks, plan, max_repairs = retrieval.context_chunks(), retrieval.plan(), cfg.max_repair_iters

        generate_and_test(cfg, log, corpus=corpus, request=request, chunks=chunks, index=index,
                          max_repairs=max_repairs, plan=plan)
    except RefusedError as e:
        log.write(status="refused", error=str(e))
        raise
    except Exception as e:
        # Failed runs are logged too, so the lab can count them.
        log.write(status="error", error=f"{type(e).__name__}: {e}")
        raise
    log.write(status="ok")
    return log, retrieval

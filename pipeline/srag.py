"""
Standard RAG (srag): retrieve once, then generate once. This is the baseline the
agentic pipeline is compared against, and it does what start.py did.
"""

from .api_check import check_code, extract_code_blocks
from .api_index import build_api_index
from .config import Config
from .embeddings import make_embedder
from .llm import call_claude
from .logger import RunLog
from .prompts import build_request, load_prompt, render
from .retrieve import format_context, retrieve


def run_srag(cfg: Config, *, corpus: str, concepts: str, dataset_description: str = "") -> RunLog:
    log = RunLog(cfg, mode="srag", corpus=corpus, concepts=concepts, dataset_description=dataset_description)
    try:
        index = build_api_index(cfg, corpus)
        log.record["api_index_version"] = index.version
        request = build_request(concepts, dataset_description)

        # 1. Retrieve: the whole request is the search query (no decomposition in srag).
        chunks = retrieve(cfg, corpus, request, make_embedder(cfg))
        log.add_iteration({
            "iteration": 0, "concepts": None, "critique": None,
            "retrievals": [{"concept": None, "query": request, "chunks": [c.log_entry() for c in chunks]}],
        })

        # 2. Generate: retrieved chunks go into the system prompt.
        template, version = load_prompt("srag_system")
        log.add_prompt("srag_system", version)
        system = render(template, context=format_context(chunks))
        result = call_claude(cfg, role="srag_generate", system=system, messages=request)
        log.add_llm_call(result)
        log.record["generated"]["raw_text"] = result.text  # split into parts from Phase 3

        # 3. Hallucination check on the code blocks in the reply.
        log.add_api_check(check_code(extract_code_blocks(result.text), index))

        log.write(status="refused" if result.stop_reason == "refusal" else "ok")
    except Exception as e:
        # Failed runs are logged too, so the lab can count them.
        log.write(status="error", error=f"{type(e).__name__}: {e}")
        raise
    return log

"""
Run logging: one JSON object per run, appended as one line to runs/runs.jsonl.

"JSON Lines" means each line is a complete JSON record, so the lab can load the
file with pandas.read_json(path, lines=True) and every run is one row.

Every record has the same keys, whatever the mode. Fields a mode doesn't use stay
null (e.g. srag has no critique), so all conditions can be compared side by side.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import PART_ORDERS, Config

SCHEMA_VERSION = 4   # 2: per-concept retrievals, retrieval_stop, api_index_version, api_check
                     # 3: attempts, outcome
                     # 4: every mode generates + tests; parts_order, context; outcome passed|failed;
                     #    generated.raw_text removed


class RunLog:
    def __init__(self, cfg: Config, *, mode: str, corpus: str, concepts: str, dataset_description: str):
        self.cfg = cfg
        self._started = time.perf_counter()
        self.record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": None,              # ok | refused | error
            "error": None,
            "mode": mode,
            "order": cfg.order,
            "parts_order": list(PART_ORDERS[cfg.order]),   # the order the generator wrote the parts in
            "dataset": corpus,
            "requested_concepts": concepts,
            "dataset_description": dataset_description,
            "decomposed_concepts": None,        # agentic: the final concept plan
            "iterations": [],                   # one entry per retrieval loop iteration (see add_iteration)
            "retrieval_stop": None,             # agentic: accepted | cap_reached | no_api_concepts
            "api_index_version": None,          # hash of the corpus's API name list
            "apis_used": None,                  # real API names the generated code calls
            "hallucinated_apis": None,          # names it calls that aren't in the corpus
            "api_check": None,                  # full check result: suggestions, parse errors
            "context": None,                    # docs given to the generator (see retrieve.apply_budget); None in nc
            "generated": {"description": None, "solution": None, "tests": None},   # final version
            "execution": None,                  # final test run: passed, counts, output, error text
            "outcome": None,                    # passed (tests pass, no unknown APIs) | failed
            "repair_iterations": 0,             # repair calls made (always 0 in nc and srag)
            "attempts": [],                     # every version: files + api check + test run
            "prompts": {},                      # prompt name -> version hash
            "llm_calls": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "wall_clock_s": None,
            "config": cfg.to_dict(),
        }

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    def add_prompt(self, name: str, version: str) -> None:
        self.record["prompts"][name] = version

    def add_llm_call(self, result) -> None:
        self.record["llm_calls"].append(result.log_entry())
        self.record["total_input_tokens"] += result.input_tokens
        self.record["total_output_tokens"] += result.output_tokens

    def add_iteration(self, entry: dict) -> None:
        """
        Record one retrieval iteration:
          {"iteration": 0,
           "concepts": [...] if a new concept plan started here, else None,
           "retrievals": [{"concept": ..., "query": ..., "chunks": [{"id", "source", "score"}]}],
           "critique": {"action", "reason", "concepts": [verdict per concept]} or None (srag)}
        nc has no retrieval, so its list stays empty.
        """
        self.record["iterations"].append(entry)

    def add_api_check(self, check) -> None:
        self.record["apis_used"] = check.apis_used
        self.record["hallucinated_apis"] = check.hallucinated
        self.record["api_check"] = check.to_dict()

    def add_attempt(self, attempt: int, assignment, check, execution) -> None:
        """Record one generated version; the latest one is also the run's final result."""
        self.record["attempts"].append({
            "attempt": attempt,   # 0 = first generation, 1+ = after each repair
            **assignment.to_dict(),
            "api_check": check.to_dict(),
            "execution": execution.to_dict(),
        })
        self.record["generated"].update(assignment.to_dict())
        self.record["execution"] = execution.to_dict()
        self.add_api_check(check)

    def write(self, status: str, error: str | None = None) -> Path:
        self.record["status"] = status
        self.record["error"] = error
        self.record["wall_clock_s"] = round(time.perf_counter() - self._started, 3)
        path = Path(self.cfg.runs_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False) + "\n")
        return path

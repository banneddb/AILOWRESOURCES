"""
Single entry point for the pipeline. Every experiment condition runs through here.

  python run.py ingest bridges                 # read data/bridges/ into the vector DB
  python run.py ingest bridges --reset         # rebuild that collection from scratch
  python run.py search bridges "add an edge"   # show what retrieval returns (no Claude call)
  python run.py apis jes                       # show the corpus's API index (no Claude call)
  python run.py apis jes --check my_code.py    # run the hallucination check on a file
  python run.py generate --corpus bridges --concepts "loops, dictionaries" \
                         --dataset-description "USGS earthquake data"
  python run.py generate --mode agentic --corpus jes --concepts "loops, lists"
  python run.py generate --mode nc --order tests_first --budget 100 --corpus jes --concepts "loops"

Settings come from config.toml. Override any of them for one run with --set:
  python run.py --set top_k=10 --set chunk_size=300 generate ...
"""

import argparse
import sys

from dotenv import load_dotenv

from pipeline.config import ConfigError, load_config, parse_override


def cmd_ingest(cfg, args):
    from pipeline.ingest import ingest_corpus
    print(f"Ingesting data/{args.corpus}/ ...")
    summary = ingest_corpus(cfg, args.corpus, reset=args.reset)
    print(f"Done: {summary['chunks']} chunks from {summary['files']} files "
          f"into {summary['collection']} ({summary['collection_size']} chunks total)")


def cmd_search(cfg, args):
    from pipeline.embeddings import make_embedder
    from pipeline.retrieve import retrieve
    for chunk in retrieve(cfg, args.corpus, args.query, make_embedder(cfg), k=args.k):
        preview = chunk.text[:200].replace("\n", " ")
        print(f"{chunk.score:.3f}  {chunk.id}\n       {preview}...\n")


def cmd_apis(cfg, args):
    from pipeline.api_index import build_api_index
    index = build_api_index(cfg, args.corpus)
    if args.check:
        from pipeline.api_check import check_code
        with open(args.check, encoding="utf-8") as f:
            print(check_code(f.read(), index).to_dict())
        return
    print(f"{len(index)} API names in {args.corpus} (index version {index.version}):")
    print(", ".join(sorted(index.names)))


def print_retrieval(result):
    print(f"Retrieval stopped: {result.stop}\n")
    for c in result.concepts:
        v = result.verdicts.get(c["name"])
        status = v["verdict"] if v else "no retrieval (plain Python)"
        print(f"- {c['name']}: {status}")
        if c["api_need"]:
            print(f"    needs: {c['api_need']}")
        if v:
            print(f"    APIs: {', '.join(v['apis_verified']) or '-'}"
                  + (f"  (not in corpus: {', '.join(v['apis_not_in_corpus'])})" if v["apis_not_in_corpus"] else ""))
            print(f"    critic: {v['reason']}")
    print(f"\nChunks for the generator: {[ch.id for ch in result.context_chunks()]}")


def cmd_generate(cfg, args):
    from pipeline.modes import run
    log, retrieval = run(cfg, corpus=args.corpus, concepts=args.concepts,
                         dataset_description=args.dataset_description)
    r = log.record
    if retrieval:
        print_retrieval(retrieval)
    c = r["context"]
    docs = (f"{len(c['chunks_given'])} of {c['chunks_available']} chunks, {c['lines_given']} lines "
            f"(budget: {c['budget_lines'] or 'none'})") if c else "none (nc mode)"
    print(f"\nDocs given to the generator: {docs}\nParts written in this order: {', '.join(r['parts_order'])}")
    print(f"\n{'=' * 70}\n{r['generated']['description']}\n\n--- solution.py ---\n{r['generated']['solution']}"
          f"\n\n--- test_solution.py ---\n{r['generated']['tests']}\n{'=' * 70}")
    for a in r["attempts"]:
        print(f"attempt {a['attempt']}: tests {a['execution']['summary']}; "
              f"not in corpus: {a['api_check']['hallucinated'] or 'none'}")
    print(f"Outcome: {r['outcome']} after {r['repair_iterations']} repairs. APIs used: {r['apis_used']}")
    print(f"\n[run {r['run_id']} | mode {r['mode']} | status {r['status']} | {len(r['llm_calls'])} Claude calls | "
          f"{r['total_input_tokens']} in / {r['total_output_tokens']} out tokens | "
          f"{r['wall_clock_s']}s | logged to {cfg.runs_file}]")


def main():
    parser = argparse.ArgumentParser(description="Assignment generation pipeline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value for this run (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk + embed data/<corpus>/ into the vector DB")
    p.add_argument("corpus")
    p.add_argument("--reset", action="store_true", help="delete the collection first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="show the top-k chunks for a query")
    p.add_argument("corpus")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("apis", help="show a corpus's API index, or check a code file against it")
    p.add_argument("corpus")
    p.add_argument("--check", metavar="FILE", help="Python file to run the hallucination check on")
    p.set_defaults(func=cmd_apis)

    p = sub.add_parser("generate", help="generate an assignment (mode from config or --mode)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--concepts", required=True, help='e.g. "loops, dictionaries"')
    p.add_argument("--dataset-description", default="")
    p.add_argument("--mode", help="nc | srag | agentic (shortcut for --set mode=...)")
    p.add_argument("--order", help="description_first | solution_first | tests_first (shortcut for --set order=...)")
    p.add_argument("--budget", type=int, help="max lines of docs for the generator, 0 = no limit "
                                              "(shortcut for --set context_budget_lines=...)")
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a codepage that can't print everything
    load_dotenv()  # puts ANTHROPIC_API_KEY from .env into the environment; never printed

    try:
        overrides = dict(parse_override(s) for s in args.set)
        for flag, key in (("mode", "mode"), ("order", "order"), ("budget", "context_budget_lines")):
            if getattr(args, flag, None) is not None:
                overrides[key] = getattr(args, flag)
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        sys.exit(str(e))

    try:
        args.func(cfg, args)
    except (FileNotFoundError, LookupError, RuntimeError) as e:   # includes AgentError
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()

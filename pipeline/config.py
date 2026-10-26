"""
Load config.toml into a Config object and check that the values make sense.

The TOML file groups settings into sections ([chunking], [retrieval], ...) for
readability. Every key name is unique, so we flatten them into one object:
cfg.chunk_size, cfg.top_k, and so on.
"""

import tomllib
from dataclasses import asdict, dataclass, fields

MODES = ("nc", "srag", "agentic")
ORDERS = ("description_first", "solution_first", "tests_first")
EMBEDDING_PROVIDERS = ("sentence-transformers", "ollama")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
LANGUAGES = ("python",)            # languages with a code runner (see pipeline/runners/)
LIBRARY_RUNTIMES = ("fake", "installed")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    # [paths]
    data_dir: str
    chroma_dir: str
    runs_file: str
    # [ingest]
    extensions: tuple
    # [corpus_include]: corpus name -> list of glob patterns
    corpus_include: dict
    # [chunking]
    chunk_size: int
    overlap: int
    # [embedding]
    provider: str
    model: str
    ollama_url: str
    # [retrieval]
    top_k: int
    max_concepts: int
    max_queries_per_concept: int
    # [generation]
    generation_model: str
    max_tokens: int
    effort: str
    # [loops]
    max_critique_loops: int
    max_repair_iters: int
    json_retries: int
    # [execution]
    language: str
    python_executable: str
    timeout_s: int
    max_output_chars: int
    # [library_runtime]: corpus name -> "fake" | "installed"
    library_runtime: dict
    # [experiment]
    mode: str
    order: str
    context_budget_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


# The TOML uses "model" in both [embedding] and [generation]; rename one so the
# flat names stay unique.
_RENAMES = {("generation", "model"): "generation_model"}

# Sections whose keys are names you choose (like corpus names), kept whole as one dict.
_TABLE_SECTIONS = ("corpus_include", "library_runtime")


def parse_override(text: str) -> tuple[str, object]:
    """Turn 'top_k=10' or 'mode=agentic' into ('top_k', 10) / ('mode', 'agentic')."""
    if "=" not in text:
        raise ConfigError(f"--set expects key=value, got {text!r}")
    key, value = (part.strip() for part in text.split("=", 1))
    try:
        parsed = tomllib.loads(f"v = {value}")["v"]   # numbers, true/false, "quoted strings"
    except tomllib.TOMLDecodeError:
        parsed = value                                # bare words like agentic stay strings
    return key, parsed


def load_config(path: str = "config.toml", overrides: dict | None = None) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    flat = {section: raw.pop(section, {}) for section in _TABLE_SECTIONS}
    for section, values in raw.items():
        for key, value in values.items():
            name = _RENAMES.get((section, key), key)
            if name in flat:
                raise ConfigError(f"Key {name!r} appears in more than one section")
            flat[name] = value
    flat.update(overrides or {})

    known = {f.name for f in fields(Config)}
    unknown = sorted(set(flat) - known)
    missing = sorted(known - set(flat))
    if unknown:
        raise ConfigError(f"Unknown config keys (typo?): {unknown}")
    if missing:
        raise ConfigError(f"Missing config keys: {missing}")

    flat["extensions"] = tuple(flat["extensions"])
    for f in fields(Config):
        if not isinstance(flat[f.name], f.type):
            raise ConfigError(
                f"{f.name} should be {f.type.__name__}, got {flat[f.name]!r}"
            )

    cfg = Config(**flat)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    errors = []
    if cfg.chunk_size <= 0:
        errors.append("chunk_size must be > 0")
    # The chunker steps forward by (chunk_size - overlap) words each time; if that
    # step is 0 or negative it never reaches the end of the text.
    if not 0 <= cfg.overlap < cfg.chunk_size:
        errors.append(f"overlap must be >= 0 and < chunk_size ({cfg.chunk_size}), got {cfg.overlap}")
    for name in ("top_k", "max_concepts", "max_queries_per_concept", "timeout_s", "max_output_chars"):
        if getattr(cfg, name) < 1:
            errors.append(f"{name} must be >= 1")
    if cfg.max_tokens < 1:
        errors.append("max_tokens must be >= 1")
    if min(cfg.max_critique_loops, cfg.max_repair_iters, cfg.json_retries) < 0:
        errors.append("loop caps and json_retries must be >= 0")
    if cfg.context_budget_lines < 0:
        errors.append("context_budget_lines must be >= 0 (0 = no limit)")
    for corpus, patterns in cfg.corpus_include.items():
        if not (isinstance(patterns, list) and patterns and all(isinstance(p, str) for p in patterns)):
            errors.append(f"corpus_include.{corpus} must be a non-empty list of glob strings")
    for corpus, runtime in cfg.library_runtime.items():
        if runtime not in LIBRARY_RUNTIMES:
            errors.append(f"library_runtime.{corpus} must be one of {LIBRARY_RUNTIMES}, got {runtime!r}")
    for name, value, allowed in [
        ("mode", cfg.mode, MODES),
        ("order", cfg.order, ORDERS),
        ("provider", cfg.provider, EMBEDDING_PROVIDERS),
        ("effort", cfg.effort, EFFORTS),
        ("language", cfg.language, LANGUAGES),
    ]:
        if value not in allowed:
            errors.append(f"{name} must be one of {allowed}, got {value!r}")
    if errors:
        raise ConfigError("Invalid config:\n  " + "\n  ".join(errors))

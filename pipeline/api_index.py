"""
The API index: every function, class, method, and module name that really
exists in a corpus. The hallucination check compares generated code against it.

It's rebuilt from the corpus files each run (it takes about a second), so it
always matches the files config.toml says belong to the corpus. Names are found
with simple patterns, not a full parser, because the corpora mix Python 2 (JES
is Jython), Java, and Markdown docs.
"""

import difflib
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .ingest import iter_doc_files, read_doc

PY_DEF = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)
JAVA_TYPE = re.compile(r"\b(?:class|interface|enum)\s+([A-Za-z_]\w*)")
# A Java method declaration: "<modifiers> <return type> name(<params>) [throws ...] {"
JAVA_METHOD = re.compile(
    r"^\s*(?:[\w<>\[\],.?]+\s+)+([A-Za-z_]\w*)\s*\([^;{)]*\)\s*(?:throws\s+[\w.,\s]+)?\{",
    re.MULTILINE,
)
JAVA_KEYWORDS = {"if", "for", "while", "switch", "catch", "synchronized", "return", "new", "else", "try"}
MD_CALL = re.compile(r"`([A-Za-z_][\w.]*)\s*\(")      # `name(...)` in Markdown/text docs
HTML_CALL = re.compile(r"\b([A-Za-z_][\w.]*)\(")      # name(...) in generated HTML docs


@dataclass
class ApiIndex:
    corpus: str
    names: dict   # name -> list of files it appears in

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

    @property
    def version(self) -> str:
        """Short hash of the name list, logged so a run can be tied to the exact index it used."""
        return hashlib.sha256("\n".join(sorted(self.names)).encode()).hexdigest()[:12]

    def suggest(self, name: str, n: int = 3) -> list[str]:
        """Real names that look like a hallucinated one (fed to the repair step in Phase 3)."""
        return difflib.get_close_matches(name, self.names.keys(), n=n, cutoff=0.75)


def names_in_file(path: Path, text: str) -> set[str]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        found = set(PY_DEF.findall(text)) | {path.stem}   # the module name itself, for "import media"
    elif suffix == ".java":
        found = set(JAVA_TYPE.findall(text)) | (set(JAVA_METHOD.findall(text)) - JAVA_KEYWORDS)
    elif suffix in (".md", ".rst", ".txt"):
        found = set(MD_CALL.findall(text))
    elif suffix in (".html", ".htm"):
        found = set(HTML_CALL.findall(text))
    else:
        # TODO(.lj): pull function definitions out of Less-Java programs once we
        # generate Less-Java code. For now its api_reference.md lists the built-ins.
        found = set()
    # "obj.method" -> "method"; skip private names like _helper and __init__
    return {n.split(".")[-1] for n in found if not n.split(".")[-1].startswith("_")}


def build_api_index(cfg: Config, corpus: str) -> ApiIndex:
    corpus_dir = Path(cfg.data_dir) / corpus
    names: dict[str, list[str]] = {}
    for path in iter_doc_files(corpus_dir, cfg.extensions, cfg.corpus_include.get(corpus)):
        source = path.relative_to(corpus_dir).as_posix()
        for name in names_in_file(path, read_doc(path)):
            names.setdefault(name, []).append(source)
    if not names:
        raise LookupError(f"Found no API names in data/{corpus}/. Is the corpus empty?")
    return ApiIndex(corpus=corpus, names=names)

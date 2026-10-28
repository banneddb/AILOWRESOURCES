"""
Code runners: run a generated solution + tests in a separate process and report
what happened. One runner per language, all returning the same ExecResult.
"""

from ..config import Config
from .base import ExecResult
from .python_runner import PythonRunner


def get_runner(cfg: Config):
    if cfg.language == "python":
        return PythonRunner(cfg)
    # TODO(java): JavaRunner - write Solution.java + SolutionTest.java, compile with javac,
    #   run JUnit; needed for BRIDGES-Java assignments.
    # TODO(less-java): LessJavaRunner - compile the .lj file with the Less-Java compiler
    #   (data/less-java), then run it; Less-Java has built-in test syntax.
    raise NotImplementedError(f"No runner for language {cfg.language!r} yet")


__all__ = ["ExecResult", "get_runner"]

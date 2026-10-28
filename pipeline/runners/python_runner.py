"""
Run a Python assignment: solution.py + test_solution.py, using unittest.
"""

import re
import subprocess
import sys

from ..config import Config
from .base import ExecResult, keep_tail, run_in_tempdir


class PythonRunner:
    language = "python"
    solution_file = "solution.py"
    test_file = "test_solution.py"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.python = cfg.python_executable or sys.executable
        self._version = None

    @property
    def version(self) -> str:
        """e.g. "3.12.5", told to the generator so it writes code this Python can run."""
        if self._version is None:
            out = subprocess.run([self.python, "-c", "import sys; print(sys.version.split()[0])"],
                                 capture_output=True, text=True, timeout=30)
            self._version = out.stdout.strip() or "3"
        return self._version

    def run(self, solution: str, tests: str) -> ExecResult:
        # -E: ignore PYTHON* environment variables; -B: don't write .pyc files
        command = [self.python, "-E", "-B", "-m", "unittest", "-v", self.test_file.removesuffix(".py")]
        returncode, stdout, stderr, timed_out, seconds = run_in_tempdir(
            {self.solution_file: solution, self.test_file: tests}, command, self.cfg.timeout_s)

        # unittest prints its summary to stderr, e.g. "Ran 5 tests" then "OK" or "FAILED (failures=1, errors=2)"
        ran = re.search(r"^Ran (\d+) tests?", stderr, re.MULTILINE)
        failures = re.search(r"failures=(\d+)", stderr)
        errors = re.search(r"errors=(\d+)", stderr)
        tests_run = int(ran.group(1)) if ran else None
        return ExecResult(
            passed=returncode == 0 and not timed_out and bool(tests_run),   # "Ran 0 tests" is not a pass
            returncode=returncode,
            timed_out=timed_out,
            tests_run=tests_run,
            failures=int(failures.group(1)) if failures else (0 if ran else None),
            errors=int(errors.group(1)) if errors else (0 if ran else None),
            duration_s=round(seconds, 3),
            stdout=keep_tail(stdout, self.cfg.max_output_chars),
            stderr=keep_tail(stderr, self.cfg.max_output_chars),
        )

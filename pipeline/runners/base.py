"""
Shared pieces for all runners: the result type and a "run this command on these
files in a throwaway folder" helper.

What the sandbox does and doesn't protect against:
  - Generated code runs in its own process, never inside the pipeline, so a crash
    or infinite loop can't take the pipeline down. A timeout stops runaway code.
  - It runs in a fresh temporary folder that is deleted afterwards.
  - It gets a minimal set of environment variables, so it can't read your API key.
  - It is NOT a security sandbox: the code runs as you, so it could still read
    your files or use the network. Fine for code Claude wrote for a classroom
    exercise; a container (e.g. Docker) would be needed for untrusted code.
"""

import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Environment variables the child process gets. Everything else (including
# ANTHROPIC_API_KEY) is left out.
_SAFE_ENV = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LANG", "LC_ALL")


@dataclass
class ExecResult:
    passed: bool
    returncode: int | None
    timed_out: bool
    tests_run: int | None
    failures: int | None
    errors: int | None
    duration_s: float
    stdout: str
    stderr: str

    def summary(self) -> str:
        if self.timed_out:
            return "timed out"
        if self.passed:
            return f"passed ({self.tests_run} tests)"
        if self.tests_run:
            return f"failed: {self.tests_run} tests, {self.failures or 0} failures, {self.errors or 0} errors"
        return f"failed before any test ran (exit code {self.returncode})"

    def to_dict(self) -> dict:
        return asdict(self) | {"summary": self.summary()}


def keep_tail(text: str, limit: int) -> str:
    """Keep the end of long output: that's where test summaries and tracebacks are."""
    return text if len(text) <= limit else f"[... {len(text) - limit} characters cut ...]\n" + text[-limit:]


def run_in_tempdir(files: dict[str, str], command: list[str], timeout_s: int):
    """Write `files` into a new temp folder, run `command` there. Returns (returncode, stdout, stderr, timed_out, seconds)."""
    env = {k: os.environ[k] for k in _SAFE_ENV if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="assignment_run_") as folder:
        for name, content in files.items():
            Path(folder, name).write_text(content, encoding="utf-8")
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout_s,
                                  stdin=subprocess.DEVNULL)   # no stdin: code waiting for input() fails fast
            return proc.returncode, proc.stdout, proc.stderr, False, time.perf_counter() - started
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return None, out, err, True, time.perf_counter() - started

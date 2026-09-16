"""Parallel headless LLM calls via the claude CLI.

This container has no API key and no SDK credentials; the claude CLI is
authenticated, so judge calls go through `claude -p`. Haiku is deliberate:
the experiment needs a cheap mid-tier judge with dynamic range (a judge
that reads perfectly leaves no room for the difficulty signal), and the
run makes thousands of calls.
"""

import os
import subprocess
import concurrent.futures
import tempfile

MODEL = "claude-sonnet-5"   # probe: clean TIE floor on identical pairs; haiku had none
WORKERS = 8    # shared LXC host: be a good citizen; more workers does not help under contention

# The judge must not inherit any project CLAUDE.md (the workspace one says
# "reply in Chinese", which contaminated a smoke test). Run from a neutral
# directory outside the repo.
NEUTRAL_CWD = tempfile.mkdtemp(prefix="judge-")


def call(prompt, model=MODEL, retries=3, timeout=90):
    for _ in range(retries):
        try:
            r = subprocess.run(["claude", "-p", "--model", model],
                               input=prompt, capture_output=True, text=True, timeout=timeout,
                               cwd=NEUTRAL_CWD)
            out = (r.stdout or "").strip()
            if r.returncode == 0 and out:
                return out
        except subprocess.TimeoutExpired:
            pass
    return ""


def pmap(prompts, model=MODEL, workers=WORKERS):
    """Order-preserving parallel map over prompts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: call(p, model), prompts))


if __name__ == "__main__":
    import time
    t0 = time.time()
    outs = pmap(['Reply with exactly one word: SAME or DIFFERENT. Text A: "x is 1". Text B: "x is 2".'] * 8)
    print(f"8 parallel calls in {time.time()-t0:.1f}s -> {outs}")

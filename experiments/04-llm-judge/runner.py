"""Run the registered blocks when the shared host is actually usable.

This container sits on a contended LXC fleet host: /proc/loadavg reports the
whole machine, and judge throughput has ranged from 0.8 s/call (quiet) to
12 s/call (load 300+). So the gate is measured throughput, not load: probe
with two calls, and only spend a block when the host can carry it. Every
block is cached, so this is resumable and nothing is ever redone.
"""
import json, os, subprocess, sys, time
import er
from llm import pmap

BLOCKS = list(er.REG_BLOCKS)
MAX_S_PER_CALL = 4.0
WAIT = 300
DEADLINE = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 10800)


def probe():
    b = json.load(open("blocks.json"))["blocks"][9]      # spare block, never registered
    ps = [er.sd_prompt(it["question"], it["a"], it["b"]) for it in b["battery"][:2]]
    t0 = time.time()
    outs = pmap(ps)
    per = (time.time() - t0) / len(ps)
    return per, all(outs)


while time.time() < DEADLINE:
    todo = [b for b in BLOCKS if not os.path.exists(f"results/block{b}.json")]
    if not todo:
        print("all registered blocks done", flush=True)
        break
    per, ok = probe()
    if not ok or per > MAX_S_PER_CALL:
        print(f"host busy ({per:.1f}s/call), waiting {WAIT}s; {len(todo)} blocks left", flush=True)
        time.sleep(WAIT)
        continue
    b = todo[0]
    print(f"host ok ({per:.1f}s/call) -> running block {b} ({len(todo)} left)", flush=True)
    r = subprocess.run(["../../.venv/bin/python", "-u", "er.py", "registered", str(b)],
                       capture_output=True, text=True, timeout=3600)
    print(r.stdout[-400:] if r.stdout else r.stderr[-400:], flush=True)

todo = [b for b in BLOCKS if not os.path.exists(f"results/block{b}.json")]
print(f"runner finished; remaining blocks: {todo}", flush=True)
if not todo:
    r = subprocess.run(["../../.venv/bin/python", "-u", "er.py", "registered"],
                       capture_output=True, text=True, timeout=3600)
    print(r.stdout, flush=True)

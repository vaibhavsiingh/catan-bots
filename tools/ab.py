"""Paired A/B test through the harness (harness.match.play_chunk), in parallel chunks.

Every arm plays the games `python -m harness.evaluate --games N --seed0 S` plays (game i on board S + i//2, seats
alternating, the same dice from both seats), so arms are compared board by board. Each chunk runs in its
own process: the harness starts agents as daemon spawn children, which a multiprocessing Pool cannot host.

usage (from any directory):
  python tools/ab.py OUT NAME=agent.py[@KEY=VAL,KEY=VAL] ... [--b OPP_AGENT]
      [--games 2000] [--seed0 990000] [--par 7] [--chunk 100] [--noclock] [--log DIR]

- KEY=VAL pairs become environment variables of that arm's chunks (e.g. BB_LA_K=8). They reach both
  agents' processes, so don't use them when the opponent is one of our own agents.
- Finished chunks stay in OUT/NAME/, so a rerun with more games plays only the new ones.
- The first arm is the baseline of the paired differences.
- --noclock plays both seats without the CPU clock (the harness's clocked=False: cpu_left is None). Parallel
  chunks inflate measured CPU (9 chunks on 8 cores: about 1.8x), which pushes games over the clock, so
  compare strength with --noclock and check the CPU of the final candidate in a serial run.
- --log DIR sets BB_LOG=DIR/<seed0>_<lo> for each chunk's agents (agents/lookahead writes training data there;
  experiments/kit_exit.py adds the results).
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def play(agent, opp, seed0, lo, hi, out, clock, log):
    """worker: games lo..hi-1 of an evaluate run with this seed0; records to `out`"""
    if log:
        os.environ["BB_LOG"] = os.path.join(log, f"{seed0}_{lo}")
    sys.path.insert(0, ROOT)
    from harness import match
    idx = list(range(lo, hi))
    records, _ = match.play_chunk(agent, opp, [seed0 + i // 2 for i in idx], idx, clock_a=clock, clock_b=clock)
    for r in records:
        r["seed0"] = seed0
    with open(out + ".tmp", "w") as f:
        json.dump(records, f)
    os.replace(out + ".tmp", out)


def parse_arm(spec):
    name, rest = spec.split("=", 1)
    path, _, kv = rest.partition("@")
    env = dict(p.split("=", 1) for p in re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", kv)) if kv else {}  # values may hold commas
    return name, os.path.abspath(path), env


def load(out, name):
    """every finished game of one arm, keyed by (seed0, game index)"""
    d, games = os.path.join(out, name), {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and f != "arm.json":
            for r in json.load(open(os.path.join(d, f))):
                games[(r["seed0"], r["game"])] = r
    return games


def summary(out, names):
    arms = {n: load(out, n) for n in names}
    base = names[0]
    for n in names:
        g = list(arms[n].values())
        if not g:
            continue
        k = len(g)
        w = sum(r["winner"] == "a" for r in g)
        p = w / k
        cpu = [r["a"]["cpu_s"] for r in g]
        bad = sum(r["a"]["errors"] + r["a"]["timeouts"] + r["a"]["crashes"] for r in g)
        opp_bad = sum(r["b"]["timeouts"] + r["b"]["crashes"] + (r["b"]["over"] is not None) for r in g)
        line = (f"{n:14s} {p:.3f} ± {1.96 * math.sqrt(p * (1 - p) / k):.3f} ({w}/{k})  "
                f"CPU mean {1000 * sum(cpu) / k:4.0f} worst {1000 * max(cpu):4.0f} ms  "
                f"over {sum(r['a']['over'] is not None for r in g)}  faults {bad}"
                + (f"  OPPONENT played random moves in {opp_bad} games" if opp_bad else ""))
        if n != base:  # paired: per board, (our wins with this arm - with the baseline) / 2
            common = arms[n].keys() & arms[base].keys()
            boards = {}
            for key in common:
                b = (key[0], key[1] // 2)
                boards[b] = boards.get(b, 0) + ((arms[n][key]["winner"] == "a") - (arms[base][key]["winner"] == "a")) / 2
            d = [v for b, v in boards.items() if sum((b[0], 2 * b[1] + s) in common for s in (0, 1)) == 2]
            if len(d) > 1:
                m = sum(d) / len(d)
                sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
                line += f"   vs {base}: {100 * m:+.1f} ± {196 * sd / math.sqrt(len(d)):.1f} pts ({len(d)} boards)"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--b", default=os.path.join(ROOT, "agents", "f", "agent.py"))
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=990000)
    ap.add_argument("--par", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--noclock", action="store_true")
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    assert a.chunk % 2 == 0, "chunks must hold whole boards (both seats)"
    a.out = os.path.abspath(a.out)  # workers run in ROOT
    a.log = os.path.abspath(a.log) if a.log else ""
    if a.log:
        os.makedirs(a.log, exist_ok=True)
    opp = os.path.abspath(a.b)
    arms = [parse_arm(s) for s in a.arms]
    jobs = []
    for name, path, env in arms:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        spec = dict(path=os.path.relpath(path, ROOT), env=env, opp=os.path.relpath(opp, ROOT), clock=not a.noclock)
        meta = os.path.join(d, "arm.json")
        if os.path.exists(meta):
            assert json.load(open(meta)) == spec, f"{name} already ran with other settings: {meta}"
        json.dump(spec, open(meta, "w"), indent=1)
        for lo in range(0, a.games, a.chunk):
            hi = min(lo + a.chunk, a.games)
            f = os.path.join(d, f"{a.seed0}_{lo}_{hi}.json")
            if not os.path.exists(f):
                jobs.append((lo, name, path, env, hi, f))
    jobs.sort(key=lambda j: j[0])  # arms interleaved: partial results stay comparable
    running, done, t0 = [], 0, time.time()
    print(f"{len(jobs)} chunks to play, {a.par} at a time", flush=True)
    while jobs or running:
        while jobs and len(running) < a.par:
            lo, name, path, env, hi, f = jobs.pop(0)
            e = dict(os.environ, PYTHONHASHSEED="0", **env)
            if a.noclock:  # no CPU clock: a move may take up to 2 minutes; only a stuck agent is cut off
                e.setdefault("CATAN_CALL_TIMEOUT_S", "120")
            cmd = ["nice", "-n", "19", sys.executable, os.path.abspath(__file__), "--worker", path, opp, str(a.seed0), str(lo), str(hi), f, str(int(not a.noclock)), a.log]
            running.append((subprocess.Popen(cmd, env=e, cwd=ROOT), name, lo))
        time.sleep(2)
        for p in [r for r in running if r[0].poll() is not None]:
            running.remove(p)
            done += 1
            if p[0].returncode:
                print(f"chunk {p[1]} @{p[2]} failed (exit {p[0].returncode})", flush=True)
            print(f"[{time.strftime('%H:%M')}] {done} chunks done, {time.time() - t0:.0f}s", flush=True)
    summary(a.out, [n for n, _, _ in arms])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        play(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8] == "1", sys.argv[9])
    else:
        main()

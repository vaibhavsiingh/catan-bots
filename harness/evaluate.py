"""Play two agents against each other and report the result.

    python -m harness.evaluate --a agents/distill/agent.py --b agents/f/agent.py --games 300

Each agent runs in its own process and sees only a masked copy of the game: the opponent's unplayed
development cards are redrawn at random, and so is the deck. Every limit is enforced, not just
measured; see harness/match.py for the values and the reasons.

Seats alternate. Each board is played twice, once from each seat, with the same dice both times.
"""
import argparse
import sys
import time

from harness._patch import ensure_hash_seed
from harness import match


def summarise(records, side="a"):
    """Win rate and limit counters for one side of a list of game records from match.play_chunk."""
    n = len(records)
    me = [r[side] for r in records]
    wins = sum(r["winner"] == side for r in records)
    rate = wins / n
    cpu = [m["cpu_s"] for m in me]
    return dict(games=n, wins=wins, win_rate=rate, ci95=1.96 * (rate * (1 - rate) / n) ** 0.5,
                no_winner=sum(r["winner"] is None for r in records),
                mean_cpu_s=sum(cpu) / n, worst_cpu_s=max(cpu),
                decisions=sum(m["decisions"] for m in me),
                games_over_budget=sum(m["over"] is not None for m in me),
                random_moves=sum(m["random_moves"] for m in me),
                errors=sum(m["errors"] for m in me),
                timeouts=sum(m["timeouts"] for m in me),
                crashes=sum(m["crashes"] for m in me))


def run(path_a, path_b, games, seed0=990000, vps=10, key=match.DEFAULT_KEY, quiet=False):
    t0 = time.time()
    seeds = [seed0 + i // 2 for i in range(games)]
    records, info = match.play_chunk(path_a, path_b, seeds, list(range(games)), key=key, vps=vps)
    res = summarise(records)
    res.update(init_s=info["a_init_s"], wall_s=time.time() - t0)
    if not quiet:
        lim = (f"{match.CPU_FLAT_CAP_S*1000:.0f} ms per game" if match.CPU_FLAT_CAP_S is not None
               else f"{match.CPU_BASE_S*1000:g} ms + {match.CPU_PER_DECISION_S*1000:g} ms per decision, "
                    f"at most {match.CPU_MAX_S*1000:g} ms")
        print(f"A = {path_a}")
        print(f"B = {path_b}")
        print(f"A win rate          {res['win_rate']:.4f} +-{res['ci95']:.4f}   "
              f"({res['wins']}/{games})")
        print(f"A mean CPU / game   {res['mean_cpu_s']*1000:8.1f} ms   (limit {lim})")
        print(f"A worst CPU / game  {res['worst_cpu_s']*1000:8.1f} ms")
        print(f"A games over budget {res['games_over_budget']:8d}      "
              f"(random moves played after the limit: {res['random_moves']})")
        print(f"A errors            {res['errors']:8d}      illegal index or exception, of "
              f"{res['decisions']} decisions")
        print(f"A timeouts/crashes  {res['timeouts']:3d} / {res['crashes']:<3d}   (a silent agent is "
              f"killed after {match.CALL_TIMEOUT_S:.0f} s)")
        print(f"A load time         {res['init_s']:8.2f} s    (limit {match.INIT_TIMEOUT_S:.0f} s)")
        print(f"wall clock          {res['wall_s']:8.1f} s")
    return res


def main(argv=None):
    ensure_hash_seed()          # may restart this process with PYTHONHASHSEED=0
    p = argparse.ArgumentParser(description="Play two Catan agents and report the result.")
    p.add_argument("--a", required=True, help="path to your agent file")
    p.add_argument("--b", required=True, help="path to the opponent agent file")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=990000, help="first board seed")
    p.add_argument("--vps", type=int, default=10)
    args = p.parse_args(argv)
    run(args.a, args.b, args.games, args.seed0, args.vps)


if __name__ == "__main__":
    sys.exit(main())

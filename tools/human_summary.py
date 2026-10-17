"""Results of the bot against people, from the games the web app records (web/server.py, play mode).

usage: python tools/human_summary.py [games/human] [--verify]

Counts finished games (someone reached 10 VP), the bot's win rate with a 95% interval, games per player
(players are random browser ids, nothing more) and abandoned games. --verify replays every game from its
log in the engine and checks that the replay reaches the recorded winner.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=os.path.join(ROOT, "games", "human"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    games = [json.load(open(f)) | {"_file": f} for f in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    done = [g for g in games if g.get("winner")]
    print(f"{len(games)} games recorded, {len(done)} finished, {len(games) - len(done)} abandoned")
    if done:
        w = sum(g["winner"] == "bot" for g in done)
        p = w / len(done)
        ci = 1.96 * math.sqrt(p * (1 - p) / len(done))
        print(f"bot won {w}/{len(done)} = {100 * p:.1f}% ± {100 * ci:.1f}")
        for first in ("us", "them"):
            sub = [g for g in done if g["first"] == first]
            if sub:
                print(f"  {'human' if first == 'us' else 'bot'} placed first: bot won {sum(g['winner'] == 'bot' for g in sub)}/{len(sub)}")
        print(f"  mean length {sum(g['turns'] for g in done) / len(done):.0f} turns")
        players = Counter(g.get("player") or "?" for g in done)
        print(f"  {len(players)} players; games per player: " + ", ".join(f"{n}" for _, n in players.most_common(10))
              + (" ..." if len(players) > 10 else ""))
    if a.verify:
        import advise as A
        bad = 0
        for g in games:
            try:
                game = A.new_game(g)
                for e in g["log"]:
                    A.apply(game, e)
                wc = game.winning_color()
                got = None if wc is None else ("human" if wc == A.US else "bot")
                if got != g.get("winner"):
                    raise AssertionError(f"replay winner {got}, recorded {g.get('winner')}")
            except Exception as ex:
                bad += 1
                print(f"  {os.path.basename(g['_file'])}: {ex}")
        print(f"verified: {len(games) - bad}/{len(games)} replay to the recorded result")


if __name__ == "__main__":
    main()

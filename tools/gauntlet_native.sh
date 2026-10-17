#!/bin/bash
# Opponents that are too slow for the harness's 3 s per decision, played natively with no limit
# (tools/native_match.py: the opponent sees the true game). 100 games each, distill and F (reference).
#   MCTS(100), AlphaBeta depth 3, and the HexMachina players if HEXMACHINA=path/to/hexmachina-repro
#   (github.com/OguroHayato/hexmachina-repro, with results/<run>_best_player.py copied to hexmachina/best_<run>.py).
cd "$(dirname "$0")/.."
OUT=${1:-results/gauntlet}
PY=${PYTHON:-python}
PAR=${PAR:-15}
match() {  # name, env...
    name=$1; shift
    for a in distill:agents/distill/agent.py F:agents/f/agent.py; do
        echo "== native $name, ${a%%:*}"
        env "$@" $PY tools/native_match.py $OUT/native_${name}_${a%%:*} ${a#*:} --games 100 --chunk 4 --par $PAR | tail -1
    done
}
match M CAT=M
match AB3 CAT=AB CAT_DEPTH=3
if [ -n "$HEXMACHINA" ]; then
    for p in e2g e7b e5bg; do match hex_$p PYTHONPATH=$HEXMACHINA CAT=PY CAT_CLASS=hexmachina.best_$p:FooPlayer; done
fi

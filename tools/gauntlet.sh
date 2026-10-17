#!/bin/bash
# The distill bot and F (reference) against every catanatron bot and our other agents: 300 games each,
# clock off, boards 550000+, through the harness; then the full-information AlphaBeta runs
# (tools/native_match.py: the opponent sees the true game).
# usage: tools/gauntlet.sh [OUT=results/gauntlet]     (PYTHON=... to choose the interpreter)
cd "$(dirname "$0")/.."
OUT=${1:-results/gauntlet}
PY=${PYTHON:-python}
DISTILL=agents/distill/agent.py
F=agents/f/agent.py
PAR=${PAR:-14}
run() {  # name, opponent agent, env for both arms
    echo "== $1"
    $PY tools/ab.py $OUT/$1 distill=$DISTILL${3:+@$3} F=$F${3:+@$3} --b $2 --games 300 --seed0 550000 --par $PAR --chunk 20 --noclock | grep -v "chunks done"
}
for c in R W VP F; do run cat_$c tools/cat_agent.py CAT=$c; done
run cat_AB2 tools/cat_agent.py CAT=AB,CAT_DEPTH=2
for s in ppo heuristic ppo_search lookahead; do run $s agents/$s/agent.py; done
for a in distill:$DISTILL F:$F; do
    echo "== native AB2, ${a%%:*}"
    CAT=AB CAT_DEPTH=2 $PY tools/native_match.py $OUT/native_AB2_${a%%:*} ${a#*:} --games 300 --par $PAR | tail -1
done

#!/bin/bash
# The runs behind the paper's tables, at the lowest CPU priority on PAR cores (default 8).
#   1. headline: distill vs F with the CPU clock (1000 games), distill and F vs AlphaBeta-2 (1000, no clock)
#   2. round robin of every agent, ratings on one scale (400 games per pair)
#   3. ablations of distill, one part removed at a time, against F, AlphaBeta-2 and lookahead (600 games each)
# Everything resumes: rerun the script and only unfinished chunks are played.
# usage: tools/paper_runs.sh [OUT=results]      (PYTHON=..., PAR=...)
cd "$(dirname "$0")/.."
OUT=${1:-results}
PY=${PYTHON:-python}
PAR=${PAR:-8}
D=agents/distill/agent.py
F=agents/f/agent.py
AB2="tools/cat_agent.py"
stamp() { echo "[$(date +%H:%M)] $*"; }

stamp "1a headline: distill vs F, clocked"
$PY tools/ab.py $OUT/headline/vs_F_clocked distill=$D --b $F --games 1000 --seed0 900000 --par $((PAR * 3 / 4)) --chunk 20 | grep -v "chunks done"
stamp "1b headline: distill and F vs AlphaBeta-2, no clock"
$PY tools/ab.py $OUT/headline/vs_AB2 distill=$D@CAT=AB,CAT_DEPTH=2 F=$F@CAT=AB,CAT_DEPTH=2 --b $AB2 --games 1000 --seed0 900000 --par $PAR --chunk 20 --noclock | grep -v "chunks done"

stamp "2 round robin"
$PY tools/roundrobin.py $OUT/roundrobin distill=$D lookahead=agents/lookahead/agent.py ppo=agents/ppo/agent.py \
    ppo_search=agents/ppo_search/agent.py heuristic=agents/heuristic/agent.py F=$F AB2=$AB2@CAT=AB,CAT_DEPTH=2 \
    ${EXTRA_AGENTS} --games 400 --seed0 700000 --par $PAR --noclock | grep -v "chunks done"

ARMS=(
  full=$D
  no_judge=$D@DISTILL_MARGIN=inf
  outcome_judge=$D@DISTILL_JUDGE=outcome
  no_dev_term=$D@DISTILL_A2_WEIGHTS='{"dev_owned": 0}'
  no_planner=$D@DISTILL_CONFIG='{"planner": false}'
  no_chains=$D@DISTILL_CONFIG='{"trade_chain": false, "road_plan": false}'
  no_win_check=$D@DISTILL_CONFIG='{"win_check": false}'
  no_knight_rule=$D@DISTILL_CONFIG='{"knight_rule": false}'
  no_robber_rule=$D@DISTILL_CONFIG='{"robber_rule": false}'
  no_opening_model=$D@DISTILL_OPENING=0
  rules_off=$D@DISTILL_MARGIN=inf,DISTILL_A2_WEIGHTS='{"dev_owned": 0}',DISTILL_CONFIG='{"planner": false, "win_check": false, "knight_rule": false, "robber_rule": false, "trade_chain": false, "road_plan": false}'
)
for opp in F AB2 lookahead; do
    case $opp in
        F) B=$F; X="" ;;
        AB2) B=$AB2; X="CAT=AB,CAT_DEPTH=2" ;;
        lookahead) B=agents/lookahead/agent.py; X="" ;;
    esac
    arms=()
    for a in "${ARMS[@]}"; do
        if [ -n "$X" ]; then case $a in *@*) a="$a,$X" ;; *) a="$a@$X" ;; esac; fi
        arms+=("$a")
    done
    stamp "3 ablation vs $opp"
    $PY tools/ab.py $OUT/ablation/vs_$opp "${arms[@]}" --b $B --games 600 --seed0 800000 --par $PAR --chunk 20 --noclock | grep -v "chunks done"
done
stamp "done"

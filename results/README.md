# Results

`gauntlet/` holds the game records behind the table in the main README: one folder per opponent, with one
sub-folder per agent rated against it (`distill`, `F`), or `native_<opponent>_<agent>` for runs where the
opponent plays on the true game (`tools/native_match.py`). Each `arm.json` records the settings of its run.

    python tools/summarize.py results/gauntlet

Boards 550000 onward, both seats per board, clock off (`--noclock`) so that parallel runs compare strength,
not CPU contention.

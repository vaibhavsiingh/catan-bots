"""The agent's side of a match: one OS process per agent, talking to the harness over a pipe.

WHY A SEPARATE PROCESS. In its own process the agent owns nothing the harness depends on. It
receives a COPY of the game in which the hidden information has been resampled, it has its own
random-number generator, and its CPU is read by the harness from the operating system, from
outside, where nothing it does can reach.

THIS FILE RUNS INSIDE THE AGENT'S PROCESS, so the agent can change any of it at run time: patch
these functions, send other replies, skip the reseeding. The harness is built so that this gains
nothing. It never trusts a value from this side: it measures the agent's CPU itself, it reads each
reply as plain JSON and accepts only an integer index of a legal move, and everything the agent
receives was already masked before it left the harness.
"""
import importlib.util
import json
import operator
import os
import sys
import time
import traceback


def _load_agent(path):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))     # let the agent import its own helper modules
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Agent"):
        raise AttributeError(f"{path} defines no class named Agent")
    return module.Agent


#: Seed of the global `random` in every agent process, at start and after every restart.
AGENT_RANDOM_SEED = 20261013


def _reply(conn, *msg):
    """Replies are plain JSON bytes. The harness parses them as data and accepts only an integer
    index, so nothing the agent returns is ever turned back into a Python object on the harness side.
    """
    conn.send_bytes(json.dumps(msg).encode())


def _as_index(x):
    if isinstance(x, bool):
        raise TypeError("bool is not an action index")
    return int(operator.index(x))      # numpy integers pass; floats and strings raise TypeError


def serve(conn, agent_path, memory_bytes=None, uid=None, scratch=None):
    """Run inside the child. Protocol:
        harness -> agent   ("go",)                          once, before the agent loads
        agent   -> harness ["ready", 0, load_seconds]  or  ["init_error", 0, traceback_text]
        harness -> agent   ("newgame", seed)                before every game, no reply
        harness -> agent   ("decide", seq, view_game, actions, color, cpu_left)   pickled
        agent   -> harness ["ok", seq, index]  or  ["error", seq, traceback_text]   JSON bytes

    seq numbers the decisions, so the harness can tell a late answer to an earlier decision from
    the answer to the current one.
    """
    if uid is not None and os.getuid() == 0:
        # Run as root with a per-agent uid: give up root before any agent code exists in this process.
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        # Output is discarded in that setup: errors come back in the replies, and an agent that
        # printed without end must not fill a disk. Normal runs keep their prints.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    if scratch is not None:
        # The private directory the harness made for this process: libraries that cache under ~ or
        # write temporary files use it, never root's home or a shared /tmp.
        os.environ["HOME"] = os.environ["TMPDIR"] = scratch
    # Run from the agent's own directory, in every setup, so a relative path to a
    # weights file behaves the same in both places.
    agent_path = os.path.abspath(agent_path)           # before the chdir, against the caller's cwd
    os.chdir(os.path.dirname(agent_path))
    sandboxed = os.environ.get("CATAN_SANDBOX") == "1"
    if memory_bytes and sys.platform.startswith("linux") and not sandboxed:
        # Enforced on Linux. macOS accepts the call and ignores it, so local runs are unbounded.
        # Under the sandbox the cgroup's memory.max does this instead. `resource` exists on Unix
        # only, so it is imported here and never on Windows.
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Wait until the harness has finished setting this process up (under the sandbox: moved
    # it into its cgroup), so that everything the agent does happens inside it.
    try:
        conn.recv()
    except EOFError:
        return

    import harness  # noqa: F401  applies the deterministic trade-order fix in this process too
    from harness.obs import Obs

    # The engine draws from Python's global `random` when a move has a random outcome, for example
    # the card a robber steals. Searching agents execute such moves on their copies, so without a
    # fixed seed here the same agent plays different games on every run and no game can be replayed.
    # This generator is the agent's own; the dice are drawn in the harness process, not here.
    import random
    random.seed(AGENT_RANDOM_SEED)

    t0 = time.monotonic()
    try:
        agent = _load_agent(agent_path)()
    except BaseException:
        _reply(conn, "init_error", 0, traceback.format_exc(limit=8)[-4000:])
        return
    _reply(conn, "ready", 0, time.monotonic() - t0)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        if msg[0] == "newgame":
            # Every game starts the agent's `random` from that game's own seed, so a game played
            # alone replays exactly as it played in the middle of a run.
            random.seed(msg[1])
            continue
        _, seq, view, actions, color, cpu_left = msg
        try:
            obs = Obs(view, color)
            obs.cpu_left = cpu_left
            idx = _as_index(agent.decide(obs, actions))
            _reply(conn, "ok", seq, idx)
        except BaseException:
            _reply(conn, "error", seq, traceback.format_exc(limit=8)[-4000:])

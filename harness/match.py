"""Play games between isolated agent processes, with every limit enforced by the harness.

The limits are defined once, below, so the documentation and the code cannot drift apart.
"""
import hashlib
import hmac
import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
import time

import psutil

import harness  # noqa: F401  deterministic trade-order fix, in the harness process
from harness import host, sandbox
from harness.obs import masked_copy
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer

#: The clock. CPU time of the agent, every thread and child process it starts, as the operating
#: system reports it. The allowance grows with the game, like a chess clock with an increment:
#:
#:     allowance after k decisions  =  min(CPU_BASE_S  +  k * CPU_PER_DECISION_S,  CPU_MAX_S)
#:
#: Only real decisions count: a move with one legal action is played by the harness and never shown
#: to the agent. Games run from about 20 to 160 decisions per player, so a flat cap would cut off an
#: agent in exactly the long, close games. The increment gives a long game proportionally more time,
#: and the base pays for the opening placement, which has the most candidates and comes first. At
#: F's mean of about 56 decisions a game the allowance is about 1 s. The ceiling, reached at 96
#: decisions, bounds what one game can cost: at most 2 x CPU_MAX_S of agent
#: CPU per game. Once the allowance is spent, the harness plays uniformly random legal moves for the
#: rest of that game.
CPU_BASE_S = 0.300
CPU_PER_DECISION_S = 0.0125
CPU_MAX_S = 1.5
#: Set to a number to use a flat per-game cap instead of the clock above.
CPU_FLAT_CAP_S = None
#: Time spent WAITING on the agent per game, beyond its CPU allowance. A sleeping or blocked agent
#: uses no CPU, so a CPU cap alone never stops it; this bounds how long one agent can hold up a match.
WALL_SLACK_S = 3.0
#: One decision. An agent silent this long is killed and restarted, and random moves finish the game.
#: CATAN_CALL_TIMEOUT_S overrides it (tools/ab.py and tools/roundrobin.py set 120 s for --noclock runs, so
#: slow searchers are judged on their moves, and only an agent that is stuck gets cut off).
CALL_TIMEOUT_S = float(os.environ.get("CATAN_CALL_TIMEOUT_S", "3.0"))
#: While an agent thinks, the harness reads its CPU this often and stops waiting the moment the
#: allowance is spent, so no decision can overspend the clock by more than about this much.
POLL_S = 0.005
#: Importing and constructing the agent, once per process. Not charged to any game.
INIT_TIMEOUT_S = 60.0
#: Memory limit for the agent process. Enforced on Linux; macOS ignores it.
MEMORY_BYTES = 2 * 1024 ** 3

#: Longest reply the harness reads from an agent. A reply is an index or a short traceback.
MAX_REPLY_BYTES = 16 * 1024

COLORS = [Color.RED, Color.BLUE]
#: Default dice key. Pass a private key to play_chunk if dice must not be predictable from a board.
DEFAULT_KEY = b"catan-kit-public-default-key"


def dice_seed(board_seed, pair_index, key=DEFAULT_KEY):
    """Dice are decoupled from the board.

    Game(seed=s) seeds the board AND the dice. An agent that knows s and sees the move history can
    replay the engine and predict every roll, and the board is enough to brute-force a small or
    sequential s. Re-seeding after the board is built, from a keyed hash, keeps boards pairable across
    seats and opponents while making the dice unpredictable without the key. Both seats of one board
    share a pair_index, so they also share the dice.
    """
    msg = f"{board_seed}:{pair_index}".encode()
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], "big")


class AgentProcess:
    """One agent running in its own OS process."""

    def __init__(self, path, memory_bytes=MEMORY_BYTES, uid=None):
        self.path, self.memory_bytes, self.uid = path, memory_bytes, uid
        self.init_s = None
        self.cg = None
        self.scratch = None
        self.busy = False          # still working on a decision the harness stopped waiting for
        self.seq = 0               # numbers every decision this process is asked, across games
        self.start()

    def start(self):
        if self.uid is not None and hasattr(os, "geteuid") and os.geteuid() == 0:
            # Run as root with per-agent uids: a fresh private directory, made here as root and given to
            # the agent's user, becomes its HOME and TMPDIR. Shared directories such as /tmp are not
            # writable by agents there, so two agents cannot pass messages through files, and
            # nothing an agent writes outlives its process. CATAN_SCRATCH_ROOT puts it on a memory
            # filesystem, where what it writes counts against its memory limit, not our disk.
            self.scratch = tempfile.mkdtemp(prefix=f"catan-{self.uid}-",
                                            dir=os.environ.get("CATAN_SCRATCH_ROOT"))
            os.chown(self.scratch, self.uid, self.uid)
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=host.serve,
                                args=(child, self.path, self.memory_bytes, self.uid, self.scratch),
                                daemon=True)
        self.proc.start()
        child.close()
        self.ps = psutil.Process(self.proc.pid)
        if sandbox.ENABLED:
            self.cg = sandbox.Cgroup(self.memory_bytes)
            self.cg.add(self.proc.pid)
        try:
            self.conn.send(("go",))
        except (BrokenPipeError, OSError):
            self.kill()
            raise RuntimeError(f"{self.path}: agent process died while starting "
                               f"(exit code {self.proc.exitcode})")
        if not self.conn.poll(INIT_TIMEOUT_S):
            self.kill()
            raise RuntimeError(f"{self.path}: agent did not finish loading in {INIT_TIMEOUT_S:.0f} s")
        msg = self._recv()
        if msg is None:
            self.kill()
            raise RuntimeError(f"{self.path}: agent process died while loading "
                               f"(exit code {self.proc.exitcode})")
        if msg[0] == "init_error":
            self.kill()
            raise RuntimeError(f"{self.path}: agent failed to load\n{msg[2]}")
        self.init_s = msg[2]
        self.busy = False

    def cpu(self, full=False):
        """CPU of the agent, user plus system, as the operating system reports it.

        Read from the harness, so nothing the agent does inside its own process can falsify it.
        Under the sandbox: the cgroup's total, which includes every thread and process the agent
        started. Otherwise: the agent process and its threads, plus children it has waited for, and
        with full=True also live child processes. Finding those scans every process on the machine,
        so it runs at the start and end of each game, not at every decision.
        """
        if self.cg is not None:
            try:
                return self.cg.cpu()
            except OSError:
                return float("inf")
        try:
            t = self.ps.cpu_times()
            total = t.user + t.system + t.children_user + t.children_system
            if full:
                for c in self.ps.children(recursive=True):
                    try:
                        ct = c.cpu_times()
                        total += ct.user + ct.system
                    except psutil.Error:
                        pass
            return total
        except psutil.Error:
            return float("inf")

    def _recv(self):
        """Read one reply as JSON data. Returns None if the process is gone or the reply is not a
        short JSON list of the form [status, seq, value]."""
        try:
            msg = json.loads(self.conn.recv_bytes(MAX_REPLY_BYTES))
        except (EOFError, OSError, ValueError):
            return None
        if not (isinstance(msg, list) and len(msg) == 3 and isinstance(msg[0], str)):
            return None
        return msg

    def decide(self, view, actions, color, cpu_left=None, cpu_deadline=None):
        """Ask for one decision. Returns (status, value); status is "ok", "error", "timeout",
        "crash", or "over" when the agent's CPU passed cpu_deadline before it answered."""
        self.seq += 1
        seq = self.seq
        try:
            self.conn.send(("decide", seq, view, actions, color, cpu_left))
        except (BrokenPipeError, OSError):
            return "crash", None
        t_end = time.monotonic() + CALL_TIMEOUT_S
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                self.busy = True
                return "timeout", None
            if self.conn.poll(min(POLL_S, left)):
                msg = self._recv()
                if msg is None:
                    return "crash", None
                if msg[1] != seq:              # a late answer to a decision already given up on
                    continue
                return msg[0], msg[2]
            if cpu_deadline is not None and self.cpu() > cpu_deadline:
                self.busy = True
                return "over", None

    def new_game(self, seed):
        """Reseed the agent process's global `random` for the game about to start."""
        try:
            self.conn.send(("newgame", seed))
        except (BrokenPipeError, OSError):
            pass

    def settle(self):
        """Between games: let an agent that was still thinking finish, or restart it."""
        if not self.busy:
            return
        t_end = time.monotonic() + CALL_TIMEOUT_S
        while (left := t_end - time.monotonic()) > 0:
            if self.conn.poll(left):
                if self._recv() is None:
                    break
                self.busy = False
                return
        self.restart()

    def kill(self):
        if self.cg is not None:
            self.cg.kill()
            self.cg = None
        else:
            try:
                for c in self.ps.children(recursive=True):
                    c.kill()
            except psutil.Error:
                pass
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(timeout=5)
        if self.scratch is not None:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.scratch = None

    def restart(self):
        self.kill()
        self.start()

    def stop(self):
        try:
            self.conn.send(("stop",))
        except (BrokenPipeError, OSError):
            pass
        self.kill()


class Seat(RandomPlayer):
    """The engine-facing player for one agent. Enforces every limit and counts every failure."""

    def __init__(self, color, agent, clocked=True):
        super().__init__(color)
        self.agent = agent
        #: False runs a seat without the CPU and wall limits, for testing. Rated games clock both
        #: seats.
        self.clocked = clocked

    def begin_game(self, seed):
        # Private generators: the harness must never draw from the global `random`, which the
        # engine uses for the dice, so a fallback move cannot shift the dice sequence.
        self.fallback = random.Random(seed)
        self.mask_rng = random.Random(seed ^ 0x5EED)
        # The agent's own generator for this game, from a one-way hash of the game's seed: the same
        # game always gets the same stream, and the agent learns nothing about the seeds from it.
        self.agent.new_game(int.from_bytes(hashlib.sha256(f"agent:{seed}".encode()).digest()[:8], "big"))
        self.cpu0 = self.agent.cpu(full=True)
        self.last_used = 0.0
        self.wall = 0.0
        self.over = None
        self.calls = self.errors = self.timeouts = self.crashes = self.random_moves = 0

    def allowance(self):
        """CPU this agent may have used by now in this game, counting the decision about to be made."""
        if CPU_FLAT_CAP_S is not None:
            return CPU_FLAT_CAP_S
        return min(CPU_BASE_S + (self.calls + 1) * CPU_PER_DECISION_S, CPU_MAX_S)

    def _random(self, actions):
        self.random_moves += 1
        return self.fallback.choice(actions)

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        if len(actions) == 1:                       # forced: never shown to the agent
            return actions[0]
        if self.over:
            return self._random(actions)
        allow = self.allowance()                    # includes this decision's increment
        used = self.agent.cpu() - self.cpu0
        self.last_used = used
        if self.clocked and used > allow:
            self.over = "cpu"
            return self._random(actions)
        if self.clocked and self.wall > allow + WALL_SLACK_S:
            self.over = "wall"
            return self._random(actions)

        view = masked_copy(game, self.color, self.mask_rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]   # no opponent object
        view.seed = 0                                                        # no board seed
        self.calls += 1
        t0 = time.monotonic()
        status, payload = self.agent.decide(
            view, actions, self.color,
            cpu_left=(allow - used) if self.clocked else None,
            cpu_deadline=(self.cpu0 + allow) if self.clocked else None)
        self.wall += time.monotonic() - t0

        if status == "over":                        # the allowance ran out mid-decision
            self.over = "cpu"
            return self._random(actions)
        if status in ("timeout", "crash"):
            self.timeouts += status == "timeout"
            self.crashes += status == "crash"
            self.over = status
            spent = self.agent.cpu(full=True) - self.cpu0
            if spent == float("inf"):               # the process is gone: use the last reading
                spent = self.last_used
            self.agent.restart()                    # fresh process; random moves finish this game
            self.cpu0 = self.agent.cpu(full=True) - spent    # keep this game's total continuous
            return self._random(actions)
        if status != "ok":
            self.errors += 1
            return self._random(actions)
        # The reply arrived as JSON, so the index is a plain int, a float, a string or a bool. Only a
        # genuine int in range is accepted: 1.5 is not silently read as 1, and True is not read as 1.
        if type(payload) is not int or not 0 <= payload < len(actions):
            self.errors += 1
            return self._random(actions)
        return actions[payload]

    def end_game(self):
        self.cpu_used = self.agent.cpu(full=True) - self.cpu0
        self.agent.settle()

    def record(self):
        """This seat's numbers for the game just played."""
        return dict(cpu_s=round(self.cpu_used, 6), wall_s=round(self.wall, 6), decisions=self.calls,
                    over=self.over, random_moves=self.random_moves, errors=self.errors,
                    timeouts=self.timeouts, crashes=self.crashes)


def play_game(seat_a, seat_b, board_seed, game_index, key=DEFAULT_KEY, vps=10):
    """One game. Even game_index: A sits first. Odd: B sits first, on the same board and dice."""
    swap = game_index % 2 == 1
    seat_a.color = COLORS[1] if swap else COLORS[0]
    seat_b.color = COLORS[0] if swap else COLORS[1]
    order = [seat_b, seat_a] if swap else [seat_a, seat_b]
    random.seed(board_seed)
    game = Game(order, seed=board_seed, vps_to_win=vps)
    random.seed(dice_seed(board_seed, game_index // 2, key))
    s = (board_seed * 1_000_003 + game_index) & 0xFFFFFFFF
    seat_a.begin_game(s)
    seat_b.begin_game(s ^ 0xABCDEF)
    winner = game.play()
    seat_a.end_game()
    seat_b.end_game()
    return winner, game.state.num_turns


def play_chunk(path_a, path_b, board_seeds, game_indices, key=DEFAULT_KEY, vps=10,
               clock_a=True, clock_b=True, uid_a=None, uid_b=None):
    """Play a list of games between two agent files and return one record per game.

    game_indices[k] is played on board_seeds[k]. Game indices fix the seats (even: A first) and,
    with the board, the dice, so any game can be replayed alone from its record.

    Both agents load once and play every game in the list. If an agent dies, it is restarted for
    the next game and the failure is in that game's record.
    """
    sandbox.pin_self()
    a = AgentProcess(path_a, uid=uid_a)
    b = AgentProcess(path_b, uid=uid_b)
    seat_a = Seat(COLORS[0], a, clocked=clock_a)
    seat_b = Seat(COLORS[1], b, clocked=clock_b)
    out = []
    try:
        for board_seed, gi in zip(board_seeds, game_indices):
            winner, turns = play_game(seat_a, seat_b, board_seed, gi, key=key, vps=vps)
            out.append(dict(game=gi, board_seed=board_seed, a_first=gi % 2 == 0,
                            winner=None if winner is None else
                            ("a" if winner == seat_a.color else "b"),
                            turns=turns, a=seat_a.record(), b=seat_b.record()))
    finally:
        a.stop()
        b.stop()
    return out, dict(a_init_s=a.init_s, b_init_s=b.init_s)

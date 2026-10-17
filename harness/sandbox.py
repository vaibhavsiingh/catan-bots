"""Optional agent isolation with Linux cgroups. Off unless CATAN_SANDBOX=1, so it never affects a
local run: on your own machine the harness uses harness.match's process-based accounting.

Requires Linux, cgroup v2 and root. With it on, the harness gives every agent process:

    its own cgroup       CPU read from cpu.stat, which counts every thread and every process the
                         agent starts, for as long as they live. memory.max and pids.max are
                         enforced by the kernel. cgroup.kill stops everything in it at once.
    one pinned CPU       cpuset.cpus, the same CPU as the harness process playing that game. The
                         harness and both agents take turns, so a game uses exactly one core.
    its own user id      one per agent. The agent can read its own directory, the
                         Python installation and this harness, and nothing else of ours.

Reading cpu.stat is one small file read per decision. The process-based fallback has to find the
agent's child processes, which on Linux means scanning every process on the machine: measured on a
laptop, that scan was 63% of the harness's time, and it grows with the number of games in parallel.
"""
import os
import time

ENABLED = os.environ.get("CATAN_SANDBOX") == "1"
ROOT = os.environ.get("CATAN_CGROUP_ROOT", "/sys/fs/cgroup/catan-eval")
#: The CPU this game is pinned to, set per chunk by the caller.
CPU = os.environ.get("CATAN_CPU")
PIDS_MAX = 64

_count = 0


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


#: Memory kept for the operating system, the harness processes, the worker and the portal. All agent
#: cgroups together may use the rest. With every core busy and every agent at its 2 GB limit, agents
#: could ask for twice what a 384 GB machine has; with this cap the kernel kills an agent, never
#: the harness or the machine.
RESERVED_BYTES = 24 * 1024 ** 3


def setup_root():
    """Once per machine. Creates ROOT, enables the controllers its children need, and caps the
    memory of all agents together."""
    os.makedirs(ROOT, exist_ok=True)
    _write(os.path.join(ROOT, "cgroup.subtree_control"), "+cpu +cpuset +memory +pids")
    with open("/proc/meminfo") as fh:
        total = int(fh.readline().split()[1]) * 1024             # MemTotal, kB
    _write(os.path.join(ROOT, "memory.max"), str(max(total - RESERVED_BYTES, total // 2)))


def pin_self():
    """Pin the calling process (the harness for one chunk) to CPU, if one is set."""
    if CPU is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(CPU)})


class Cgroup:
    """One agent process and everything it starts."""

    def __init__(self, memory_bytes):
        global _count
        _count += 1
        self.path = os.path.join(ROOT, f"a{os.getpid()}-{_count}")
        os.mkdir(self.path)
        if memory_bytes:
            _write(self._f("memory.max"), str(memory_bytes))
            try:
                _write(self._f("memory.swap.max"), "0")
            except OSError:                       # no swap accounting on this kernel: no swap
                pass
        _write(self._f("pids.max"), str(PIDS_MAX))
        if CPU is not None:
            _write(self._f("cpuset.cpus"), CPU)

    def _f(self, name):
        return os.path.join(self.path, name)

    def add(self, pid):
        _write(self._f("cgroup.procs"), str(pid))

    def cpu(self):
        with open(self._f("cpu.stat")) as fh:
            for line in fh:
                if line.startswith("usage_usec "):
                    return int(line.split()[1]) / 1e6
        return float("inf")

    def oom_kills(self):
        try:
            with open(self._f("memory.events")) as fh:
                for line in fh:
                    if line.startswith("oom_kill "):
                        return int(line.split()[1])
        except OSError:
            pass
        return 0

    def kill(self):
        try:
            _write(self._f("cgroup.kill"), "1")
        except OSError:
            return
        for _ in range(200):                      # the kernel empties it within milliseconds
            try:
                os.rmdir(self.path)
                return
            except OSError:
                time.sleep(0.01)


def sweep(only_dead=True, prefix="a"):
    """Kill and remove agent cgroups left behind. A cgroup is named a<harness pid>-<n>; with
    only_dead it is removed only when that harness process no longer exists, so a sweep never
    touches games still being played."""
    try:
        names = os.listdir(ROOT)
    except OSError:
        return 0
    n = 0
    for d in names:
        if not d.startswith(prefix) or not os.path.isdir(os.path.join(ROOT, d)):
            continue
        if only_dead:
            try:
                os.kill(int(d[1:].split("-")[0]), 0)
                continue                          # its harness is alive: not ours to touch
            except ProcessLookupError:
                pass
            except (ValueError, PermissionError):
                continue
        cg = Cgroup.__new__(Cgroup)
        cg.path = os.path.join(ROOT, d)
        cg.kill()
        n += 1
    return n

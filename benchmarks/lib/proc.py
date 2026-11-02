"""Run a subprocess with a wall-clock timeout and a memory (RSS) cap.

The machine is shared (24 GB, other users/agents), so every solver process
started by the benchmark tooling is memory-capped:

* In-process (worker side): `limit_self(mb)` sets RLIMIT_AS / RLIMIT_DATA
  where the OS enforces them (Linux; macOS largely ignores them).
* Parent side (portable, also macOS): `run_capped` polls the child's resident
  set size via `ps` every 0.5 s and kills it when it exceeds the cap. On
  Windows it uses psutil if installed and otherwise only the wall-clock limit.
"""

from __future__ import annotations

import os
import subprocess
import time


def limit_self(mb):
    if not mb:
        return
    try:
        import resource
    except ImportError:  # Windows
        return
    lim = int(mb) * 1024 * 1024
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        r = getattr(resource, name, None)
        if r is None:
            continue
        try:
            _, hard = resource.getrlimit(r)
            soft = lim if hard == resource.RLIM_INFINITY else min(lim, hard)
            resource.setrlimit(r, (soft, hard))
        except (ValueError, OSError):
            pass


def _rss_mb_psutil(pid):
    import psutil  # optional; only used where `ps` is missing (Windows)
    try:
        p = psutil.Process(pid)
        return sum(q.memory_info().rss for q in [p] + p.children()) / (1024.0 * 1024.0)
    except psutil.Error:
        return 0.0


def _rss_mb(pid):
    """Resident set size (MB) of pid plus its direct children."""
    if os.name == "nt":
        # No `ps` / `pgrep`: use psutil if it is installed, otherwise no watchdog (the solver
        # still gets --mem-limit and enforces it itself).
        try:
            return _rss_mb_psutil(pid)
        except ImportError:
            return 0.0
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.split()
        kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True,
                              timeout=5).stdout.split()
        total = sum(int(x) for x in out if x.isdigit())
        for k in kids:
            o = subprocess.run(["ps", "-o", "rss=", "-p", k], capture_output=True, text=True,
                               timeout=5).stdout.split()
            total += sum(int(x) for x in o if x.isdigit())
        return total / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def run_capped(cmd, timeout, mem_mb=None, stdout=None, stderr=None, poll=0.5):
    """Run cmd. Returns (returncode or None, reason, wall_seconds, peak_rss_mb)
    where reason is None, 'timeout' or 'memlimit'."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    reason = None
    peak = 0.0
    while True:
        try:
            rc = p.wait(timeout=poll)
            break
        except subprocess.TimeoutExpired:
            pass
        if mem_mb:
            rss = _rss_mb(p.pid)
            peak = max(peak, rss)
            if rss > mem_mb:
                reason = "memlimit"
        if time.perf_counter() - t0 > timeout:
            reason = "timeout"
        if reason:
            if os.name != "nt":  # no pgrep on Windows; the solver starts no child processes
                try:
                    for k in subprocess.run(["pgrep", "-P", str(p.pid)], capture_output=True,
                                            text=True).stdout.split():
                        os.kill(int(k), 9)
                except (OSError, ValueError):
                    pass
            p.kill()
            p.wait()
            rc = None
            break
    return rc, reason, time.perf_counter() - t0, peak

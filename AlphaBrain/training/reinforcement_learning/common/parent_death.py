"""Linux PR_SET_PDEATHSIG helper — make this process die with its parent.

Why: when a bash launcher dies (e.g. user does `kill -9` on it, or CI/Slurm
yanks it), python child processes orphan to init (PPID=1) and survive,
keeping GPU contexts + LIBERO subprocess workers alive. The leaked workers
deadlock GPU memory until manual cleanup (or reboot, or driver reset).

Calling ``set_die_with_parent()`` early in a process's lifetime registers a
kernel hook that sends a signal (default SIGTERM) the instant the parent
process dies, so the child cleans up automatically — even when the parent
was killed via SIGKILL (untrappable from the parent side).

Used in:
  * ``trainers/train.py``        — top-level RL/pretrain entry point
  * ``envs/libero_env_worker.py`` — async-mode env subprocess
  * ``envs/libero_env_worker_fast.py`` — steplock persistent env subprocess

Linux-only (libc.prctl). On other OS this is a no-op so callers don't need
platform guards.
"""

from __future__ import annotations

import signal
import sys

_PR_SET_PDEATHSIG = 1
_HAS_PRCTL = False
_libc = None
if sys.platform.startswith("linux"):
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _HAS_PRCTL = True
    except OSError:
        _HAS_PRCTL = False


def set_die_with_parent(sig: int = signal.SIGTERM) -> bool:
    """Ask the kernel to send `sig` to this process when its parent dies.

    Returns True iff the prctl call succeeded. No-op (returns False) on
    non-Linux, or if libc/prctl is unavailable.

    Caveats:
      * Triggers when the *thread that called prctl* dies, not necessarily
        the whole process. For Python, calling this early on the main
        thread is correct.
      * If you change PPID later (e.g. via setsid + fork), this hook may
        fire when the *old* parent dies even though you've reparented.
    """
    if not _HAS_PRCTL or _libc is None:
        return False
    rc = _libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
    return rc == 0

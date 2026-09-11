"""Process-wide thread pool for LLM calls.

LLM calls run on this pool so a timeout can release the caller without waiting
for the model. A per-call ``ThreadPoolExecutor`` cannot: leaving its ``with``
block calls ``shutdown(wait=True)``, which joins the worker and re-blocks the
caller for the full duration of the very call the timeout meant to abandon.

Built on first use, never at import. Under ``gunicorn --preload`` the app is
imported once and then forked; a pool created at import time is inherited by
each child as an object whose worker threads did not survive the fork, holding
locks nothing will ever release. Creating it lazily -- and dropping the
inherited one after a fork -- gives every process its own.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading

from ..config import llm_executor_max_workers

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_llm_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the process-wide LLM executor, creating it on first use."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=llm_executor_max_workers,
                    thread_name_prefix="timbr-llm",
                )
    return _executor


def _reset_after_fork() -> None:
    """Drop the inherited executor and its lock; the next call rebuilds both."""
    global _executor, _executor_lock
    _executor = None
    _executor_lock = threading.Lock()


if hasattr(os, "register_at_fork"):  # not available on Windows
    os.register_at_fork(after_in_child=_reset_after_fork)

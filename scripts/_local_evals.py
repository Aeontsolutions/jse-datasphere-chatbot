"""Make scripts/ run THIS checkout's `evals` package.

`python scripts/run_eval.py` puts scripts/ on sys.path[0], not the repo root,
so a bare `import evals` resolves through the editable install (`pip install
-e evals/`) -- which points at whichever checkout ran that install last. With
several worktrees on one machine that is silently a different branch: a
different harness AND a different persona set, because `--personas-dir`
defaults to a path derived from the resolved package.

That has misfired twice in this repo. Once it reported "no personas matched
the filters" for personas that plainly existed; once the eval gate printed a
fail count computed by another branch's gate_stats, which looked like a real
result. The 2026-06-04 design spec hit it too and worked around it with
--personas-dir instead of fixing it.

CI never sees this -- one checkout, clean install -- which is why it survived.
It only misfires on the machines where people actually iterate.

Usage, before importing anything from `evals`:

    from _local_evals import use_local_evals
    use_local_evals()
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _assert_within(module_file: str | None, repo_root: str | Path) -> None:
    """Fail loudly if the imported `evals` came from outside this checkout.

    Prepending the repo root to sys.path normally settles it. This is the
    backstop for the cases it cannot win -- a stale `evals` already in
    sys.modules, or a .pth that injects an absolute path ahead of ours.
    Running another checkout's eval code silently is worse than not running:
    a wrong result that looks right is what cost the time here.
    """
    root = Path(repo_root).resolve()
    if module_file is None:
        raise SystemExit(
            "ERROR: the `evals` package has no __file__ -- cannot verify which "
            "checkout it came from. Refusing to run rather than risk evaluating "
            "another branch's code."
        )
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"ERROR: `evals` resolved to {resolved.parent}, which is outside this "
            f"checkout ({root}).\n"
            "This happens when `pip install -e evals/` was last run from a different "
            "worktree: you would be running that branch's harness and its personas, "
            "not this one's.\n"
            "Fix it by reinstalling from here:  pip install -e evals/"
        ) from None


def use_local_evals() -> Path:
    """Put this checkout ahead of the editable install, then verify it won.

    Returns the directory the `evals` package was loaded from. Safe to call
    more than once.
    """
    root = str(REPO_ROOT)
    if sys.path and sys.path[0] == root:
        pass
    else:
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)

    # Drop an already-imported foreign `evals` so the path change can take
    # effect; a stale entry in sys.modules would otherwise win regardless.
    stale = sys.modules.get("evals")
    stale_file = getattr(stale, "__file__", None)
    if stale is not None and stale_file is not None:
        try:
            Path(stale_file).resolve().relative_to(REPO_ROOT)
        except ValueError:
            for name in [n for n in sys.modules if n == "evals" or n.startswith("evals.")]:
                del sys.modules[name]

    import evals  # noqa: E402  -- deliberately after the sys.path fix

    _assert_within(getattr(evals, "__file__", None), REPO_ROOT)
    return Path(evals.__file__).resolve().parent

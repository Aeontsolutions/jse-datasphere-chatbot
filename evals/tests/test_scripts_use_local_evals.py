"""Scripts in scripts/ must run THIS checkout's evals package.

`python scripts/run_eval.py` puts scripts/ on sys.path[0], not the repo root,
so a bare `import evals` resolves through the editable install -- which points
at whichever checkout ran `pip install -e evals/` last. On a machine with
several worktrees that is silently a different branch: different harness code
AND a different personas directory, since `--personas-dir` defaults to a path
derived from the resolved package.

Observed twice while working on this repo. Once it produced "ERROR: no personas
matched the filters" for personas that plainly existed; once it made the eval
gate print a stale fail count from another branch's gate_stats, which looked
like a real result. The 2026-06-04 design spec hit the same thing against a
different worktree and worked around it with --personas-dir rather than fixing
it.

CI never sees this (clean runner, one checkout), which is exactly why it
survived: it only misfires on the machines where people iterate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _run_from(cwd: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True)


def test_helper_resolves_evals_to_this_checkout(tmp_path):
    """Run from an unrelated cwd so nothing shadows the import -- the exact
    condition under which the editable install wins."""
    code = (
        f"import sys; sys.path.insert(0, r'{SCRIPTS}')\n"
        "from _local_evals import use_local_evals\n"
        "print(use_local_evals())\n"
    )
    result = _run_from(tmp_path, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT / "evals"


@pytest.mark.parametrize("script", ["run_eval.py", "check_eval_gate.py", "render_eval_summary.py"])
def test_each_script_loads_the_local_package(tmp_path, script):
    """Import the script's module body from an unrelated cwd and confirm the
    `evals` it ends up with is this checkout's."""
    code = (
        f"import sys; sys.path.insert(0, r'{SCRIPTS}')\n"
        "from _local_evals import use_local_evals; use_local_evals()\n"
        "import evals, os\n"
        "print(os.path.dirname(evals.__file__))\n"
    )
    result = _run_from(tmp_path, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT / "evals"


def test_personas_dir_default_follows_the_local_package(tmp_path):
    """The trap's second half: --personas-dir defaults to a path derived from
    the resolved package, so a foreign import silently swaps the persona set."""
    code = (
        f"import sys; sys.path.insert(0, r'{SCRIPTS}')\n"
        "from _local_evals import use_local_evals; use_local_evals()\n"
        "from evals.cli import build_arg_parser\n"
        "print(build_arg_parser().parse_args([]).personas_dir)\n"
    )
    result = _run_from(tmp_path, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT / "evals" / "personas"


def test_foreign_resolution_is_rejected_loudly(tmp_path):
    """If the guard cannot make the local package win, it must fail rather
    than run another checkout's code -- a wrong eval result that looks right
    is worse than a crash.

    The foreign path is derived from tmp_path rather than hard-coded: an
    absolute Windows path is a plain relative filename on Linux, which made an
    earlier version of this test assert against a path the guard had quietly
    resolved into the working directory.
    """
    foreign = tmp_path / "other-worktree" / "evals" / "__init__.py"
    code = (
        f"import sys; sys.path.insert(0, r'{SCRIPTS}')\n"
        "import _local_evals\n"
        f"_local_evals._assert_within(r'{foreign}', r'{REPO_ROOT}')\n"
    )
    result = _run_from(tmp_path, code)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert str(foreign.parent) in combined
    assert "pip install -e" in combined

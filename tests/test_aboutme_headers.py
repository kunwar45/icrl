# ABOUTME: Enforces the CLAUDE.md rule that every tracked Python/YAML/shell file
# ABOUTME: starts with a two-line "# ABOUTME:" header. Run: pytest tests/test_aboutme_headers.py
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
# Lines allowed to precede the header: shebang, encoding cookie, Hydra package directive.
ALLOWED_PREFIX_LINES = ("#!", "# -*-", "# @package")


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if Path(line).suffix in SUFFIXES]


def missing_header(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith(ALLOWED_PREFIX_LINES):
        i += 1
    return not (
        len(lines) >= i + 2
        and lines[i].startswith("# ABOUTME:")
        and lines[i + 1].startswith("# ABOUTME:")
    )


def test_every_tracked_file_has_aboutme_header():
    offenders = sorted(
        str(p.relative_to(REPO_ROOT)) for p in tracked_files() if missing_header(p)
    )
    assert not offenders, (
        "Files missing the two-line '# ABOUTME:' header required by CLAUDE.md:\n  "
        + "\n  ".join(offenders)
    )

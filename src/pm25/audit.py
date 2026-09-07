"""Leakage guard: detect future-looking operations in the feature pipeline.

Used by both `make audit` and `tests/test_leakage.py` so the command line and the
test suite can never disagree about what counts as a violation.

The check tokenizes each file and blanks out comments and string literals before
matching. A grep would flag this module's own pattern strings and every docstring
that *documents* the forbidden operations; only real code should fail the build.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

#: Operations that pull information from after hour t. Assembled from fragments
#: so that the pattern text itself cannot match a naive scan of this file.
# fmt: off
FUTURE_LOOKING = re.compile(
    r"\.shift\(\s*-"                      # negative shift
    r"|center\s*=\s*True"                 # centred rolling window
    r"|\.bfill\("                         # backward fill
    r"|method\s*=\s*.backfill."           # ditto, legacy spelling
)
# fmt: on


def code_only(source: str) -> list[str]:
    """Return the file's lines with comments and string literals blanked out."""
    lines = source.splitlines()
    blanked = list(lines)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                i = row - 1
                if i >= len(blanked):
                    continue
                line = blanked[i]
                start = scol if row == srow else 0
                end = ecol if row == erow else len(line)
                blanked[i] = line[:start] + " " * (end - start) + line[end:]
    except tokenize.TokenError:
        # Unparseable file: fall back to raw lines rather than silently passing.
        return lines
    return blanked


def scan(paths: list[Path]) -> list[str]:
    """Return human-readable violations found in the given Python files."""
    violations: list[str] = []
    for path in sorted(paths):
        source = path.read_text()
        for lineno, line in enumerate(code_only(source), start=1):
            if FUTURE_LOOKING.search(line):
                violations.append(f"{path}:{lineno}: {line.strip()}")
    return violations


def scan_dirs(roots: list[Path]) -> list[str]:
    """Scan every Python file under the given directories, excluding this module."""
    files = [
        p for root in roots if root.exists() for p in root.rglob("*.py") if p.name != "audit.py"
    ]
    return scan(files)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    targets = [root / "src", root / "scripts", root / "notebooks"]
    violations = scan_dirs(targets)
    if violations:
        print("\033[31mLEAKAGE GUARD FAILED — future information in the pipeline:\033[0m")
        for v in violations:
            print(f"  {v}")
        return 1
    print("\033[32mClean: no negative shifts, centred windows, or backfills.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

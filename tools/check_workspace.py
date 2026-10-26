"""Check import boundaries, documentation links and distribution exclusions.

No data reads, model fitting or network access.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


def check():
    errors = []
    files = [p for folder in ("src/wade", "research", "tools", "tests") for p in (ROOT / folder).rglob("*.py")]
    for path in files:
        for module in imports(path):
            first = module.split(".")[0]
            if first == "archive":
                errors.append(f"{path.relative_to(ROOT)} imports the local archive: {module}")
            if path.is_relative_to(ROOT / "src") and first in {"research", "wade_research", "tools"}:
                errors.append(f"Production imports development code: {path.relative_to(ROOT)}: {module}")
    documents = [ROOT / n for n in ("README.md", "CONTRIBUTING.md", "ISSUES.md")]
    for folder in ("docs", "research", "tools"):
        documents.extend((ROOT / folder).rglob("*.md"))
    documents += list((ROOT / "notebooks").glob("*.qmd"))
    checked = 0
    for path in documents:
        # Ignore code examples, which may deliberately use placeholder paths.
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", text):
            link = urlsplit(target.strip("<>"))
            if link.scheme or not link.path:
                continue
            dest = (path.parent / unquote(link.path)).resolve()
            checked += 1
            if not dest.is_relative_to(ROOT) or any(dest.is_relative_to(ROOT / n) for n in (".release", "archive")):
                errors.append(f"Link outside the repository in {path.relative_to(ROOT)}: {target}")
            elif not dest.exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    excluded = re.search(r"^exclude = \[(.*?)\]", config, re.M | re.S)
    for pattern in ("archive/**", "research/**", ".release/**", "notebooks/rna100k.qmd"):
        if excluded is None or f'"{pattern}"' not in excluded[1]:
            errors.append(f"Missing distribution exclusion: {pattern}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Workspace checks passed: {len(files)} Python files, {len(documents)} documents, {checked} links.")


if __name__ == "__main__":
    check()

"""Check active source boundaries, document links and distribution exclusions.

No imports of study runners, data reads, model fitting or network access.
Local execution artifacts and historical archives are optional in a clean checkout.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


def check():
    errors = []
    files = list((ROOT / "src/wade").rglob("*.py"))
    files += list((ROOT / "research").rglob("*.py"))
    files += list((ROOT / "tools").glob("*.py"))
    files += list((ROOT / "tests").glob("*.py"))
    retired = {p.stem for p in (ROOT / "archive/development-20260923/tools").glob("*.py")}
    active_tools = {p.stem for p in (ROOT / "tools").glob("*.py")}
    for path in files:
        for module in imports(path):
            first = module.split(".")[0]
            if first == "archive" or first in retired - active_tools:
                errors.append(f"{path.relative_to(ROOT)} imports archived code: {module}")
            if path.is_relative_to(ROOT / "src") and first in {"research", "wade_research", "tools"}:
                errors.append(f"Production imports development code: {path.relative_to(ROOT)}: {module}")
    documents = [ROOT / n for n in ("README.md", "CONTRIBUTING.md", "ISSUES.md")]
    for folder in ("docs", "research", "tools"):
        documents.extend((ROOT / folder).rglob("*.md"))
    documents += list((ROOT / "notebooks").glob("*.qmd"))
    if (ROOT / "archive/README.md").exists():
        documents.append(ROOT / "archive/README.md")
    documents += list((ROOT / "archive/development-20260923").rglob("*.md"))
    checked, local_artifacts, external_files = 0, 0, 0
    for path in documents:
        # Ignore code examples, which may deliberately use placeholder paths.
        text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", text):
            link = urlsplit(target.strip("<>"))
            if link.scheme or not link.path:
                continue
            dest = (path.parent / unquote(link.path)).resolve()
            if not dest.is_relative_to(ROOT):
                external_files += 1
                continue
            if any(dest.is_relative_to(ROOT / name) for name in (".release", "archive")):
                local_artifacts += 1
                continue
            checked += 1
            if not dest.exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    config = (ROOT / "pyproject.toml").read_text()
    excluded = re.search(r"^exclude = \[(.*?)\]", config, re.M | re.S)
    for pattern in ("archive/**", "research/**", ".release/**", "notebooks/rna100k.qmd"):
        if excluded is None or f'"{pattern}"' not in excluded[1]:
            errors.append(f"Missing distribution exclusion: {pattern}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Workspace checks passed: {len(files)} Python files, {checked} source links; "
          f"{local_artifacts} optional local-artifact links, {external_files} external-file references.")


if __name__ == "__main__":
    check()

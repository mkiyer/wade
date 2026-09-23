"""Build a local artifact gallery and approval manifest from a completed render."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import re

from notebook_support import sha256


def standalone_rich_output(parts):
    """Keep the gallery's non-TeX Plotly figures usable without CDN access."""
    body = "\n".join(parts)
    runtime = ""
    if "Plotly.newPlot" in body:
        from plotly.offline import get_plotlyjs
        javascript = get_plotlyjs()
        versions = set(re.findall(r"https://cdn\.plot\.ly/plotly-([\d.]+)\.min", body))
        if any(f"plotly.js v{version}" not in javascript[:500] for version in versions):
            raise ValueError("Gallery Plotly runtime differs from the executed notebook")
        # Plotly notebook output emits both module and classic CDN loaders.
        body = re.sub(r'<script\b[^>]*src=["\']https://(?:cdn\.plot\.ly|cdnjs\.cloudflare\.com/ajax/libs/mathjax)/[^"\']+["\'][^>]*>.*?</script>',
                      "", body, flags=re.S)
        body = re.sub(r'<script\b[^>]*type=["\']module["\'][^>]*>\s*import ["\']https://cdn\.plot\.ly/[^"\']+["\']\s*</script>',
                      "", body, flags=re.S)
        runtime = "<script>" + javascript + "</script>"
    return '<!doctype html><html lang="en"><head><meta charset="utf-8">' + runtime + '</head><body>' + body + '</body></html>'


def review(build):
    build = Path(build).resolve()
    record = json.loads((build / "build.json").read_text())
    if record.get("returncode") != 0:
        raise ValueError("A review packet requires a successfully completed render")
    name = record["notebook"]
    packet = dict(demo="D", benchmark="B", examples="P", **{"reference-benchmarks": "E", "examples-breast": "C", "rna100k": "R"})[name]
    executed = build / "source/notebooks" / f"{name}.quarto_ipynb"
    cells = json.loads(executed.read_text())["cells"]
    folder = build / "review"
    folder.mkdir(exist_ok=True)
    assets = folder / "artifacts"
    assets.mkdir(exist_ok=True)
    entries, cards = [], []
    def entry(path, kind, source_cell=None, caption=None):
        identifier = f"{packet}-{len(entries)+1:03d}"
        relative = str(path.relative_to(build))
        value = dict(id=identifier, file=relative, kind=kind, source_cell=source_cell,
                     caption=caption, sha256=sha256(path), assistant_qa="pending", user_approval="pending")
        entries.append(value)
        return identifier
    for suffix in ("html", "pdf"):
        entry(build / "source/notebooks" / f"{name}.{suffix}", "complete notebook")
    for number, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        match = re.search(r"^#\| label:\s*(.+)", source, re.M)
        label = match[1].strip() if match else f"cell-{number+1}"
        cap = re.search(r"^#\| fig-cap:\s*(.+)", source, re.M)
        caption = cap[1].strip().strip("\"'") if cap else ""
        rich_html, text_outputs = [], []
        figure_output = False
        for n, output in enumerate(cell.get("outputs", []), 1):
            if output.get("output_type") == "error":
                raise ValueError(f"Error output in completed notebook: {label}")
            data = output.get("data", {})
            if "image/svg+xml" in data or "image/png" in data:
                figure_output = True
                svg = "image/svg+xml" in data
                mime, ext = ("image/svg+xml", ".svg") if svg else ("image/png", ".png")
                value = "".join(data[mime])
                path = assets / f"{label}-{n}{ext}"
                path.write_bytes(value.encode() if svg else base64.b64decode(value))
                identifier = entry(path, "figure", label, caption)
                cards.append(f'<section><h2>{identifier}: {html.escape(label)}</h2><img src="artifacts/{path.name}" alt="{html.escape(caption)}"><p>{html.escape(caption)}</p></section>')
            elif "text/html" in data:
                rich_html.append("".join(data["text/html"]))
            elif "text/plain" in data:
                text_outputs.append("".join(data["text/plain"]))
            if output.get("output_type") == "stream":
                text_outputs.append("".join(output.get("text", [])))
        if rich_html:
            figure_output = True
            path = assets / f"{label}.html"
            path.write_text(standalone_rich_output(rich_html))
            identifier = entry(path, "interactive figure / rich output", label, caption)
            cards.append(f'<section><h2>{identifier}: {html.escape(label)}</h2><p><a href="artifacts/{path.name}">Open interactive output</a></p><iframe src="artifacts/{path.name}" title="{html.escape(label)}" loading="lazy"></iframe><p>{html.escape(caption)}</p></section>')
        if cap and not figure_output and not re.search(r"^#\| eval:\s*false", source, re.M | re.I):
            raise ValueError(f"Declared figure has no rendered output: {label}")
        if text_outputs:
            path = assets / f"{label}.txt"
            path.write_text("\n".join(text_outputs))
            entry(path, "cell output", label)
    for path in sorted((build / "source/docs/figures").glob("*.png")):
        identifier = entry(path, "README illustration", "write-readme-figs")
        cards.append(f'<section><h2>{identifier}: README {path.stem}</h2><img src="../source/docs/figures/{path.name}" alt="README {path.stem}"></section>')
    for path in sorted((build / "results").rglob("*")):
        if path.is_file() and path.suffix in (".tsv", ".csv", ".json", ".npz", ".gz"):
            entry(path, "result / input / provenance export")
    exports = [item for item in entries if item["kind"] == "result / input / provenance export"]
    for item in entries:
        if item["kind"] not in ("figure", "interactive figure / rich output", "README illustration", "complete notebook"):
            continue
        # Every current notebook carries its complete numerical/provenance exports.
        supporting = exports
        if not supporting:
            raise ValueError(f"No supporting exports for {item['id']}")
        item["supporting_results"] = [e["id"] for e in supporting]
    (folder / "artifacts.json").write_text(json.dumps(dict(packet=packet, source_fingerprint=record["source_fingerprint"],
        entries=entries, user_approval="pending"), indent=2) + "\n")
    rows = []
    for item in entries:
        rows.append(f'<tr><td>{item["id"]}</td><td>{html.escape(item["kind"])}</td><td><a href="../{html.escape(item["file"])}">{html.escape(item["file"])}</a></td><td><code>{item["sha256"][:16]}</code></td><td>Pending</td></tr>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>WADE {packet}: {name} review</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#223444;background:#f3f5f7;margin:0 auto;padding:30px;max-width:1100px}}h1,h2{{line-height:1.2}}section{{padding:24px;margin:24px 0;background:white;border:1px solid #d3dbe1;border-radius:6px}}img{{width:100%;height:auto}}iframe{{width:100%;height:580px;border:0}}table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}td,th{{border:1px solid #ccd5dc;padding:8px;text-align:left;overflow-wrap:anywhere}}a{{color:#155e91}}code{{font-size:11px}}</style>
<h1>{packet}: {html.escape(name)} — candidate review</h1><p><strong>All approvals are pending.</strong> Technical inspection does not approve scientific results or figures.</p>
<p><a href="../source/notebooks/{name}.html">Complete HTML notebook</a> · <a href="../source/notebooks/{name}.pdf">PDF</a> · <a href="artifacts.json">Full hashes and approval manifest</a></p>
<p>Source fingerprint: <code>{record["source_fingerprint"]}</code></p>
{''.join(cards)}<h2>Complete artifact register</h2><table><thead><tr><th>ID</th><th>Kind</th><th>File</th><th>SHA-256 prefix</th><th>User decision</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    (folder / "index.html").write_text(page)
    print(f"{packet}: {len(entries)} artifacts; {len(cards)} visual panels; {folder / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    review(parser.parse_args().build)

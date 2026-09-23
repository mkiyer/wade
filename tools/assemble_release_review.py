"""Assemble a checksum-verified local review hub from completed notebook packets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
from notebook_support import sha256

def assemble(builds, folder):
    folder=Path(folder).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    packets, rows, artifacts = [], [], []
    for build in map(lambda p:Path(p).resolve(),builds):
        record=json.loads((build/"build.json").read_text())
        manifest=build/"review/artifacts.json"
        registry=json.loads(manifest.read_text())
        assert record["returncode"]==0
        for item in registry["entries"]:
            if sha256(build/item["file"]) != item["sha256"]:
                raise ValueError("Changed review artifact: "+str(build/item["file"]))
        qa_path=build/"review/qa.json"
        qa=json.loads(qa_path.read_text()) if qa_path.exists() else {"status":"pending"}
        packet=dict(packet=registry["packet"],notebook=record["notebook"],build=str(build),
            source_fingerprint=record["source_fingerprint"],manifest_sha256=sha256(manifest),
            artifacts=len(registry["entries"]),
            visuals=sum(e["kind"] in ("figure","interactive figure / rich output","README illustration")
                        for e in registry["entries"]),
            qa=qa,user_approval=registry["user_approval"])
        packets.append(packet)
        link=os.path.relpath(build/"review/index.html",folder)
        rows.append(f'<tr><td>{packet["packet"]}</td><td><a href="{html.escape(link)}">{html.escape(packet["notebook"])}</a></td>'
                    f'<td>{packet["visuals"]}</td><td>{packet["artifacts"]}</td><td>{html.escape(qa["status"])}</td>'
                    f'<td>{html.escape(packet["user_approval"])}</td></tr>')
        artifacts.extend(dict(packet=packet["packet"],build=str(build),**item) for item in registry["entries"])
    target=folder/"packets.json"
    if target.exists():
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(target,folder/f"packets-superseded-{stamp}.json")
    target.write_text(json.dumps(packets,indent=2)+"\n")
    (folder/"all-artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>WADE 0.2.0 review</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#243344}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ccd4dc;text-align:left}a{color:#155f98}</style>
<h1>WADE 0.2.0: current review candidates</h1>
<p><strong>All user approvals remain pending. No release has been published.</strong></p>
<p>Start with the rewritten demo (D) and the standard reference benchmarks (E).
The sample-size/pattern screen (B) includes its normalization failure and known-exposure diagnostic. The TCGA examples use full gene counts.
The private workbench (R) and its inputs remain excluded from distribution.</p>
<p>Read <a href="../../../docs/kernel-validation-roadmap.md">the scientific assessment and validation plan</a>
and <a href="../../../PLAN.md">the release queue</a>.
Each gallery links the full notebook, figures, numerical exports and content hashes.</p>
<table><thead><tr><th>Packet</th><th>Notebook</th><th>Visual panels</th><th>Artifacts</th><th>Technical QA</th><th>User approval</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p><a href="packets.json">Exact packet selection and manifest hashes</a> ·
<a href="all-artifacts.json">All registered artifacts</a></p>
<p>Revised packets supersede their earlier versions. A new rendering does not inherit approval.
Technical QA verifies execution, presentation and consistency; it does not establish calibration or scientific approval.</p></html>'''
    repo = Path(__file__).resolve().parents[1]
    page = page.replace("../../../docs/kernel-validation-roadmap.md",
                        os.path.relpath(repo / "docs/kernel-validation-roadmap.md", folder))
    page = page.replace("../../../PLAN.md", os.path.relpath(repo / "PLAN.md", folder))
    (folder/"index.html").write_text(page)
    print(f"{len(packets)} packets, {len(artifacts)} verified artifacts: {folder/'index.html'}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,default=Path(".release/0.2.0/review"))
    args=parser.parse_args()
    assemble(args.builds,args.output)

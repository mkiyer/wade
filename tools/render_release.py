"""Execute a notebook snapshot and build HTML/PDF review artifacts.

Run with the intended wade Python interpreter. Each invocation creates a
fresh directory, copies code (including the compiled kernel), and provides
its own python3 kernelspec. Quarto can execute each format separately;
verified scientific caches reuse expensive inference across presentation
builds. Nothing overwrites published figures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from notebook_support import environment, sha256


def render(name, root, config=None):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "notebooks" / f"{name}.qmd"
    if not source.exists():
        raise FileNotFoundError(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dest = root.resolve() / "builds" / f"{name}-{stamp}"
    snapshot = dest / "source"
    snapshot.mkdir(parents=True)
    for folder in ("tools", "src", "notebooks"):
        shutil.copytree(repo / folder, snapshot / folder, ignore=shutil.ignore_patterns(
            "__pycache__", "*.html", "*.pdf", "*_files", ".quarto", "*.quarto_ipynb"))
    shutil.copy2(repo / "notebooks/_mathjax-quiet.html", snapshot / "notebooks")
    shutil.copytree(repo / "docs", snapshot / "docs")
    # Historical evidence is optional local material, absent in a clean checkout.
    # Copy readable records when available, not source bundles or old runners.
    history = Path("archive/development-20260923")
    if (repo / history / "docs").exists():
        shutil.copytree(repo / history / "docs", snapshot / history / "docs")
    for relative in (history / "README.md", Path("archive/README.md")):
        if (repo / relative).exists():
            (snapshot / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, snapshot / relative)
    for file in ("README.md", "pyproject.toml"):
        shutil.copy2(repo / file, snapshot / file)
    source_hashes = {str(p.relative_to(snapshot)): sha256(p)
                     for p in sorted(snapshot.rglob("*")) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    env = os.environ.copy()
    pythonpath = [str(snapshot / "src"), str(repo / ".release/python")]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for key, folder in (("MPLCONFIGDIR", "matplotlib"), ("IPYTHONDIR", "ipython"),
                        ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        directory = dest / "cache" / folder
        directory.mkdir(parents=True)
        env[key] = str(directory)
    env["BFC_CACHE"] = str(repo / ".release/cache/bioc")
    env["QUARTO_PYTHON"] = sys.executable
    env["WADE_NOTEBOOK_OUTPUT"] = str(dest / "results")
    env["WADE_COMPUTE_CACHE"] = str(root.resolve() / "computation-cache")
    env["WADE_SOURCE_FINGERPRINT"] = fingerprint
    env.setdefault("RAYON_NUM_THREADS", "6")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    if config is not None:
        env["WADE_TCGA_CONFIG"] = str(config.resolve())
    if name in ("reference-benchmarks", "benchmark"):
        for key in ("WADE_COMPARISON_RESULTS",):
            if key not in env:
                raise ValueError(f"Set {key} to a sealed benchmark run directory")
            directory = Path(env[key]).resolve()
            if not (directory / "large/checksums.json").exists():
                raise ValueError(f"Unsealed reference run: {directory}")
            env[key] = str(directory)
    kernels = dest / "jupyter/kernels/python3"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="WADE release snapshot", language="python")))
    env["JUPYTER_PATH"] = str(dest / "jupyter")
    command = ["quarto", "render", str(snapshot / "notebooks" / source.name),
               "--to", "html,typst", "--cache", "--execute-daemon", "0",
               "-M", "keep-ipynb:true"]
    record = dict(notebook=name, created_utc=stamp, command=command,
                  source_fingerprint=fingerprint, source_hashes=source_hashes,
                  environment=environment(), user_approval="pending")
    if config is not None:
        record["config_sha256"] = sha256(config)
        shutil.copy2(config, dest / "local-config.json")
    record_path = dest / "build.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Build directory: {dest}", flush=True)
    with (dest / "render.log").open("w") as log:
        completed = subprocess.run(command, cwd=snapshot / "notebooks", env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    record["returncode"] = completed.returncode
    record["artifacts"] = {str(p.relative_to(dest)): sha256(p)
                           for p in sorted(dest.rglob("*"))
                           if p.is_file() and p.suffix in (".html", ".pdf", ".png", ".svg", ".tsv", ".csv", ".npz")}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Exit {completed.returncode}; log: {dest / 'render.log'}", flush=True)
    return completed.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", choices=["demo", "benchmark", "reference-benchmarks", "examples", "examples-breast", "rna100k"])
    parser.add_argument("--root", type=Path, default=Path(".release/0.2.0"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    raise SystemExit(render(args.notebook, args.root, args.config))

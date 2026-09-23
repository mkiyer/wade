"""Small output helpers shared by the release notebooks (not package API)."""
from __future__ import annotations

import hashlib
import ast
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values):
    values = np.asarray(values)
    if values.dtype.kind in "OUS":
        payload = json.dumps(values.tolist(), ensure_ascii=False).encode()
    else:
        payload = np.ascontiguousarray(values).tobytes()
    return hashlib.sha256(str(values.shape).encode() + str(values.dtype).encode()
                          + payload).hexdigest()


def environment():
    packages = {}
    for name in ("wade-rnaseq", "numpy", "scipy", "polars", "matplotlib", "plotly",
                 "fastexcel", "ipykernel", "jupyter-cache"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), packages=packages)


def output_dir(name):
    root = Path(os.environ.get("WADE_NOTEBOOK_OUTPUT", "./results"))
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    return out


def export_table(frame, directory, label):
    """Retain the complete displayed table, independent of display rounding."""
    frame.write_csv(Path(directory) / f"{label}.tsv", separator="\t")


def markdown_table(frame, directory=None, label=None, digits=3, display_names=None):
    if directory is not None and label is not None:
        export_table(frame, directory, label)
    if display_names:
        frame = frame.rename({k: v for k, v in display_names.items() if k in frame.columns})
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            if not np.isfinite(value):
                return "—"
            return f"{value:.{digits}g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    # Quarto's as-is output can follow progress text or another table.
    # Blank boundaries are required for Markdown to recognize a new table.
    print("\n| " + " | ".join(frame.columns) + " |")
    print("|" + "|".join("---" for _ in frame.columns) + "|")
    for row in frame.iter_rows():
        print("| " + " | ".join(map(fmt, row)) + " |")
    print()


def inference_fingerprint(extra=(), package_dir=None):
    """Hash executed inference code, native kernel and numerical dependencies."""
    import wade
    package = Path(package_dir) if package_dir is not None else Path(wade.__file__).parent
    files = list(package.glob("*.py")) + list(package.glob("_kernel*.so")) + list(package.glob("_kernel*.pyd"))
    def code_hash(path):
        path = Path(path)
        if path.suffix != ".py":
            return sha256(path)
        tree = ast.parse(path.read_text())
        # Documentation-only changes do not invalidate numerical computation.
        # Full source bytes are separately fingerprinted by the build runner.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    hashes = {p.name: code_hash(p) for p in files}
    hashes.update({str(Path(p).name): code_hash(p) for p in extra})
    hashes["numpy_version"] = np.__version__
    hashes["scipy_version"] = version("scipy")
    return hashes


def r_fingerprint():
    import subprocess
    completed = subprocess.run(["Rscript", "-e", "library(waddR); sessionInfo()"],
                               check=True, capture_output=True, text=True)
    return hashlib.sha256(completed.stdout.encode()).hexdigest()


def save_arrays(path, value):
    """Save arrays and an allowlisted dataclass tree without pickle."""
    arrays = {}
    def encode(obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                if not all(isinstance(x, str) for x in obj.flat):
                    raise TypeError("Only string object arrays are supported in result caches")
                obj = obj.astype(str)
            key = f"array_{len(arrays)}"
            arrays[key] = obj
            return {"array": key}
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj):
            return {"dataclass": type(obj).__name__, "fields":
                    {f.name: encode(getattr(obj, f.name)) for f in fields(obj)}}
        if isinstance(obj, dict):
            return {"dict": {k: encode(v) for k, v in obj.items()}}
        if isinstance(obj, (list, tuple)):
            return {"tuple" if isinstance(obj, tuple) else "list": [encode(v) for v in obj]}
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        raise TypeError(f"Unsupported cache value {type(obj)}")
    tree = encode(value)
    arrays["structure"] = np.array(json.dumps(tree))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace avoids accepting a partially written cache after a crash.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_arrays(path):
    from wade.api import WadeResult
    from wade.normalize import Normalization
    allowed = {cls.__name__: cls for cls in (WadeResult, Normalization)}
    with np.load(path, allow_pickle=False) as archive:
        def decode(obj):
            if not isinstance(obj, dict):
                return obj
            if "array" in obj:
                return archive[obj["array"]].copy(order="K")
            if "dataclass" in obj:
                cls = allowed[obj["dataclass"]]
                return cls(**{k: decode(v) for k, v in obj["fields"].items()})
            if "dict" in obj:
                return {k: decode(v) for k, v in obj["dict"].items()}
            if "tuple" in obj:
                return tuple(decode(v) for v in obj["tuple"])
            return [decode(v) for v in obj["list"]]
        return decode(json.loads(str(archive["structure"])))


def cached_computation(key, compute, *, namespace):
    root = os.environ.get("WADE_COMPUTE_CACHE")
    if not root:
        return compute()
    encoded = json.dumps(key, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    directory = Path(root) / namespace / digest
    payload, manifest = directory / "result.npz", directory / "cache.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta["key"] != key or sha256(payload) != meta["sha256"]:
            raise ValueError(f"Computation cache failed verification: {directory}")
        print(f"Verified computation cache: {namespace}/{digest[:12]}", flush=True)
        return read_arrays(payload)
    value = compute()
    save_arrays(payload, value)
    manifest.write_text(json.dumps(dict(key=key, sha256=sha256(payload)), indent=2) + "\n")
    return value


def wade_cache_key(counts, cond, **kwargs):
    """Cache identity for a public wade() call made by a notebook."""
    if hasattr(cond, "vector"):
        vector = cond.vector(kwargs.get("sample_names"))
        condition_meta = dict(case=cond.case, control=cond.control,
                              column=cond.column, dropped=list(cond.dropped))
    else:
        vector, condition_meta = np.asarray(cond), None
    def arg(value):
        from wade.normalize import Normalization
        if isinstance(value, Normalization):
            return value.to_dict()
        if isinstance(value, (list, tuple, np.ndarray)):
            return {"array_sha256": array_hash(np.asarray(value))}
        return value
    return dict(inference=inference_fingerprint(), counts=array_hash(counts),
                cond=array_hash(vector),
                condition_meta=condition_meta, parameters={k: arg(v) for k, v in kwargs.items()})


def cached_wade(counts, cond, **kwargs):
    import wade
    return cached_computation(wade_cache_key(counts, cond, **kwargs),
        lambda: wade.wade(counts, cond, **kwargs), namespace="wade")

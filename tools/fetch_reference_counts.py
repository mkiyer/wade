"""Fetch public reference count inputs, retaining URL and content provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


SOURCES = {
    "schurch-wt.tar.gz": "https://raw.githubusercontent.com/bartongroup/profDGE48/375dc0d57d9d1fa96a4245a6530e0fda34305891/Preprocessed_data/WT_countdata.tar.gz",
    "schurch-snf2.tar.gz": "https://raw.githubusercontent.com/bartongroup/profDGE48/375dc0d57d9d1fa96a4245a6530e0fda34305891/Preprocessed_data/Snf2_countdata.tar.gz",
    "schurch-exclude.txt": "https://raw.githubusercontent.com/bartongroup/profDGE48/375dc0d57d9d1fa96a4245a6530e0fda34305891/Bad_replicate_identification/exclude.lst",
    "kremer-supplements.zip": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5499207/supplementaryFiles?inlineImages=false",
    "Kremer_ODS.RDS": "https://www.cmm.in.tum.de/public/paper/OUTRIDER/supplemental_data/data/filteredODS/Kremer_ODS.RDS",
    "Skin_Not_Sun_Exposed_Suprapubic_ODS.RDS": "https://www.cmm.in.tum.de/public/paper/OUTRIDER/supplemental_data/data/filteredODS/Skin_Not_Sun_Exposed_Suprapubic_ODS.RDS",
}

SUPPLEMENTS = {"kremer-counts.tsv": 1, "kremer-patients.tsv": 3,
               "kremer-candidates.tsv": 7, "kremer-samples.tsv": 8}


def fetch_all(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "sources.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    # Reuse independently acquired author artifacts only when their source and
    # content manifests both match. Never accept an unverified file by name.
    for name, url in SOURCES.items():
        sidecar = directory / (name + ".source.json")
        if name == "kremer-supplements.zip":
            sidecar = directory / "kremer-supplements.source.json"
        if name not in previous and sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if prior["url"] == url and (directory / name).exists():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != prior["sha256"]:
                    raise ValueError(f"Source sidecar hash mismatch: {name}")
                previous[name] = prior
    manifest.write_text(json.dumps(previous, indent=2) + "\n")
    failed = []
    for name, url in SOURCES.items():
        path = directory / name
        if name in previous and path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if previous[name]["url"] != url or digest != previous[name]["sha256"]:
                raise ValueError(f"Cached public input failed verification: {name}")
            print(f"Verified {name}", flush=True)
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WADE-reference-benchmark"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                final_url = response.url
            if payload.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("Received HTML instead of a count input")
            path.write_bytes(payload)
            previous[name] = dict(url=url, final_url=final_url, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                retrieved_utc=datetime.now(timezone.utc).isoformat())
            manifest.write_text(json.dumps(previous, indent=2) + "\n")
            (directory / (name + ".source.json")).write_text(json.dumps(previous[name], indent=2) + "\n")
            print(f"Downloaded {name}: {len(payload):,} bytes", flush=True)
        except Exception as error:
            failed.append(dict(file=name, url=url, error=str(error)))
            print(f"Failed {name}: {error}", flush=True)
    archive = directory / "kremer-supplements.zip"
    if archive.exists() and "kremer-supplements.zip" in previous:
        with zipfile.ZipFile(archive) as bundle:
            for name, number in SUPPLEMENTS.items():
                suffix = f"ncomms15824-s{number}.txt"
                members = [n for n in bundle.namelist() if Path(n).name == suffix]
                if len(members) != 1:
                    raise ValueError(f"Expected one archive member {suffix}")
                payload = bundle.read(members[0])
                digest = hashlib.sha256(payload).hexdigest()
                if name in previous and digest != previous[name]["sha256"]:
                    raise ValueError(f"Changed supplementary file: {name}")
                (directory / name).write_bytes(payload)
                previous[name] = dict(archive="kremer-supplements.zip", member=members[0],
                    sha256=digest, bytes=len(payload),
                    archived_source="Original Nature links returned HTTP 404 on 2026-09-11")
        manifest.write_text(json.dumps(previous, indent=2) + "\n")
    (directory / "download-failures.json").write_text(json.dumps(failed, indent=2) + "\n")
    return not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    raise SystemExit(0 if fetch_all(parser.parse_args().directory) else 1)

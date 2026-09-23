"""Snapshot public cBioPortal annotations for the TCGA release examples.

Counts are separate local inputs. No expression values or local identifiers
are sent to the service. Responses and their request URLs/checksums are kept
together, so future notebook renders never silently refresh annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = "https://www.cbioportal.org/api"
    requests = {}
    for name, study in [("prostate", "prad_tcga_pub"),
                        ("breast", "brca_tcga_pan_can_atlas_2018"),
                        ("breast_clinical", "brca_tcga")]:
        for kind in ("SAMPLE", "PATIENT"):
            requests[f"{name}_{kind.lower()}.json"] = (
                f"{base}/studies/{study}/clinical-data?"
                f"clinicalDataType={kind}&pageSize=100000")
    requests["breast_erbb2_cna.json"] = (
        f"{base}/molecular-profiles/brca_tcga_pan_can_atlas_2018_gistic/"
        "molecular-data?sampleListId=brca_tcga_pan_can_atlas_2018_all&"
        "entrezGeneId=2064&projection=SUMMARY")
    manifest_path = out / "sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in requests.items():
        path = out / name
        if path.exists():
            entry = manifest.get(name, {})
            if (entry.get("url") != url or entry.get("sha256") !=
                    hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"Frozen snapshot failed provenance check: {path}")
            print(f"{name}: verified existing snapshot", flush=True)
            continue
        with urlopen(Request(url, headers={"Accept": "application/json",
                                          "User-Agent": "WADE-release-example"}),
                     timeout=120) as response:
            body = response.read()
        rows = json.loads(body)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty/non-list annotation response: {url}")
        path.write_bytes(body)
        manifest[name] = dict(url=url, sha256=hashlib.sha256(body).hexdigest(),
                              retrieved_utc=datetime.now(timezone.utc).isoformat(),
                              records=len(rows))
        (out / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(rows):,} records", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fetch(parser.parse_args().output)

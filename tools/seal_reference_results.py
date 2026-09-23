"""Write a complete checksum inventory of a finished reference benchmark run."""
import argparse
import json
from pathlib import Path
from notebook_support import sha256

def seal(root):
    root=Path(root).resolve()
    failures=json.loads((root/"failures.json").read_text())
    if failures:
        raise ValueError("Resolve and record benchmark failures before sealing")
    manifest=root/"checksums.json"
    files={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob("*"))
           if p.is_file() and p != manifest}
    manifest.write_text(json.dumps(dict(files=files,user_approval="pending"),indent=2)+"\n")
    print(f"Sealed {len(files)} files: {manifest}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    seal(parser.parse_args().root)

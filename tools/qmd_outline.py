"""Fence-aware Quarto outline: real markdown headers only (not R comments)."""
import re, sys

def parse(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    out, in_fence, fence_tok = [], False, None
    for i, ln in enumerate(lines, start=1):
        m = re.match(r"^(\s*)(`{3,}|~{3,})(.*)$", ln)
        if m:
            tok = m.group(2)
            if not in_fence:
                in_fence, fence_tok = True, tok[0] * 3
            elif ln.strip().startswith(fence_tok) and not m.group(3).strip():
                in_fence, fence_tok = False, None
            continue
        if in_fence:
            continue
        h = re.match(r"^(#{1,6})\s+(\S.*)$", ln)
        if h:
            out.append((i, len(h.group(1)), h.group(2).rstrip()))
    return out, len(lines)

if __name__ == "__main__":
    path = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    hdrs, n = parse(path)
    print(f"# {path}  ({n} lines)")
    for ln, lvl, txt in hdrs:
        if lo <= ln <= hi:
            print(f"{ln:>5}  {'  '*(lvl-1)}{'#'*lvl} {txt}")

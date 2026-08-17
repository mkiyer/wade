"""Extract a Quarto line range to markdown, converting {r} chunk fences to r fences.

Faithful extraction: prose, chunk options (#|) and R code are preserved
verbatim. Only the fence info string changes ({r} -> r) so the R is
syntax-highlighted as R in a plain markdown reader. Inline `r ...`
expressions and Quarto div syntax are left unresolved on purpose.
"""
import hashlib, re, sys, textwrap

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def convert(lines):
    out, in_fence = [], False
    for ln in lines:
        m = re.match(r"^(\s*)(`{3,})\{([^}]*)\}\s*$", ln)
        if m and not in_fence:
            out.append(f"{m.group(1)}{m.group(2)}r")   # ```{r ...} -> ```r
            in_fence = True
            continue
        if re.match(r"^\s*`{3,}\s*$", ln):
            out.append(ln)
            in_fence = not in_fence
            continue
        if re.match(r"^\s*`{3,}", ln) and not in_fence:
            in_fence = True
        out.append(ln)
    return out

def count_chunks(lines):
    return sum(1 for ln in lines if re.match(r"^\s*`{3,}\{[^}]*\}\s*$", ln))

def extract(src, lo, hi, dest, title, header_note, chunk_labels_note=True):
    all_lines = open(src, encoding="utf-8").read().split("\n")
    body = all_lines[lo - 1: hi]
    nchunks = count_chunks(body)
    labels = [re.sub(r"^\s*#\|\s*label:\s*", "", ln).strip()
              for ln in body if re.match(r"^\s*#\|\s*label:", ln)]
    ninline = len(re.findall(r"`r [^`]+`", "\n".join(body)))
    hdr = [
        f"# {title}",
        "",
        "> **Extracted reference material — not a specification.**",
        f"> Source: `{src}` lines {lo}–{hi} ({hi - lo + 1} lines).",
        f"> cfRNA repo git SHA `828f2f1c6a10afd182fee0e119ba3595d98b6a72` (clean tree, 2026-08-17).",
        f"> sha256 of the whole source notebook: `{sha256(src)}`.",
        f"> Retrieve the original with"
        f" `git show 828f2f1c6a10afd182fee0e119ba3595d98b6a72:{src}`.",
        ">",
        f"> Contains {nchunks} R chunk(s) and {ninline} unresolved inline `r ...`"
        " expression(s). The inline expressions are **not** evaluated here: every"
        " number they would print is absent, and no value has been guessed or"
        " substituted. Quarto div syntax (`::: {.callout-*}`), cross-references"
        " (`@sec-*`, `§`) and chunk options (`#|`) are preserved as written."
        " The only edit is the code-fence info string, `{r}` → `r`, so the R reads"
        " as R in a plain markdown viewer.",
    ]
    if header_note:
        hdr += [">", "> " + header_note.replace("\n", "\n> ")]
    if chunk_labels_note and labels:
        hdr += [">", "> Chunk labels in order: " + ", ".join(f"`{x}`" for x in labels) + "."]
    hdr += ["", "---", ""]
    open(dest, "w", encoding="utf-8").write("\n".join(hdr + convert(body)).rstrip("\n") + "\n")
    return dict(dest=dest, lines=hi - lo + 1, chunks=nchunks, inline=ninline)

if __name__ == "__main__":
    print("module")

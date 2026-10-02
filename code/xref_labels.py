"""Copy the labels of one LaTeX document's .aux file into a file another document can \\input, so the article
and Online Resource 1 can refer to each other. Only the number and page are kept, so references across the two
PDFs are printed without links."""
import argparse
from pathlib import Path

PREFIXES = ("sec:", "app:", "tab:", "fig:", "alg:", "eq:")


def brace_groups(s, n):
    out, i = [], 0
    while len(out) < n:
        i = s.index("{", i)
        depth = 0
        for j in range(i, len(s)):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                break
        out.append(s[i + 1:j])
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aux")
    ap.add_argument("out")
    a = ap.parse_args()
    lines = []
    for line in Path(a.aux).read_text().splitlines():
        if not line.startswith("\\newlabel{"):
            continue
        name, body = brace_groups(line, 2)
        if not name.startswith(PREFIXES):
            continue
        num, page = brace_groups(body, 2)
        lines.append(f"\\newlabel{{{name}}}{{{{{num}}}{{{page}}}{{}}{{}}{{}}}}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"{a.out}: {len(lines)} labels")


if __name__ == "__main__":
    main()

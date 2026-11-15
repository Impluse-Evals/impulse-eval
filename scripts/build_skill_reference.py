# SPDX-License-Identifier: Apache-2.0
"""Rebuild skills/impulse-evals/references/documentation.md from README.md, docs/ and ROADMAP.md.

Run from anywhere:

    python scripts/build_skill_reference.py

Every page becomes one "## " section, in the order below; pages under docs/ that are not listed are
appended in alphabetical order. Every heading moves one level deeper (headings inside code blocks are
left alone). Links between pages become in-file anchors: "page.md#anchor" -> "#anchor", and "page.md"
alone -> the anchor of that page's section title. Links to other repository files become their link
text. External links and image paths are left alone. The "Previous | Docs home | Next" footer of each
page is dropped. Exits 1 when a page listed in ORDER is missing.
"""
from __future__ import annotations

import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "skills" / "impulse-evals" / "references" / "documentation.md"

# (file, section title); None keeps the page's own H1 title.
ORDER = [
    ("README.md", "Overview"),
    ("docs/index.md", "Documentation home"),
    ("docs/getting-started.md", None),
    ("docs/run-for-free.md", "Run Impulse for free"),
    ("docs/concepts.md", None),
    ("docs/configuration.md", None),
    ("docs/writing-scenarios.md", None),
    ("docs/adapters.md", None),
    ("docs/voice-testing.md", None),
    ("docs/metrics.md", None),
    ("docs/compliance-packs.md", None),
    ("docs/reading-the-report.md", None),
    ("docs/cli.md", "CLI reference"),
    ("docs/ci-integration.md", None),
    ("docs/calibration.md", None),
    ("docs/cookbook.md", None),
    ("docs/troubleshooting.md", None),
    ("docs/faq.md", "FAQ"),
    ("docs/known-limitations.md", None),
    ("docs/privacy-and-security.md", None),
    ("docs/architecture.md", None),
    ("docs/contributing.md", None),
    ("docs/verified-setups.md", None),
    ("docs/changelog.md", None),
    ("ROADMAP.md", "Roadmap"),
    ("docs/skill.md", None),
]

HEADER = (
    "<!-- SPDX-License-Identifier: Apache-2.0 -->\n"
    "# Impulse Evals: complete documentation\n"
    "\n"
    "Every page of the Impulse Evals 0.1.0 documentation in one file: the README, all pages under `docs/`, "
    "and the roadmap.\n"
)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,5}) (.*)$")
SPDX = re.compile(r"^<!--\s*SPDX-License-Identifier:.*?-->\s*\n", re.S)
NAV = re.compile(r"\n---\n\n<!-- docs-nav -->\n.*\Z", re.S)
LINK = re.compile(r"(?<!!)\[((?:[^\]\\]|\\.)*)\]\(([^)\s]+)\)")
CODE_SPAN = re.compile(r"(`+[^`]*?`+)")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:|^//")


def slug(title: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
    title = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return title.replace(" ", "-")


def first_h1(page: str) -> str:
    in_fence = False
    for line in (ROOT / page).read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# "):
            return line[2:].strip()
    raise SystemExit(f"{page} has no H1 title")


def rewrite_links(line: str, page: str, anchors: dict[str, str]) -> str:
    """Rewrite the links of one prose line; code spans are left alone."""

    def fix(match: re.Match) -> str:
        text, target = match.group(1), match.group(2)
        if SCHEME.match(target) or target.startswith("#"):
            return match.group(0)
        path, _sep, fragment = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
        if resolved in anchors:
            return f"[{text}](#{fragment or anchors[resolved]})"
        return text  # another repository file: keep the link text only

    return "".join(part if part.startswith("`") else LINK.sub(fix, part) for part in CODE_SPAN.split(line))


def section(page: str, title: str, anchors: dict[str, str]) -> str:
    """One page as a "## " section: headings one level deeper, links rewritten, footer dropped."""
    text = SPDX.sub("", (ROOT / page).read_text(encoding="utf-8"), count=1)
    text = NAV.sub("\n", text)
    out: list[str] = []
    in_fence = False
    title_done = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            match = HEADING.match(line)
            if match:
                hashes, name = match.groups()
                if hashes == "#" and not title_done:
                    title_done = True
                    name = title
                line = f"#{hashes} {name}"
            line = rewrite_links(line, page, anchors)
        out.append(line)
    return "\n".join(out).strip() + "\n"


def main() -> int:
    missing = [name for name, _title in ORDER if not (ROOT / name).is_file()]
    for name in missing:
        print(f"error: {name} is listed in ORDER but does not exist", file=sys.stderr)
    if missing:
        return 1
    listed = {name for name, _title in ORDER}
    pages = list(ORDER) + [
        (path.relative_to(ROOT).as_posix(), None)
        for path in sorted(DOCS.glob("*.md"))
        if path.relative_to(ROOT).as_posix() not in listed
    ]
    titles = {name: title or first_h1(name) for name, title in pages}
    anchors = {name: slug(title) for name, title in titles.items()}

    contents = "\n".join(f"{number}. [{titles[name]}](#{anchors[name]})" for number, (name, _t) in enumerate(pages, 1))
    body = "\n---\n\n".join(section(name, titles[name], anchors) for name, _title in pages)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n## Contents\n\n" + contents + "\n\n" + body)
    print(f"Wrote {OUT.relative_to(ROOT).as_posix()} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

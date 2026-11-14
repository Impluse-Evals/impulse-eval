# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 0: no hardcoded model version strings in source code (SPEC.md section 0, rule 5)."""

import io
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "impulse_evals"
FORBIDDEN = ("claude-", "gemini-", "gpt-")
_SKIPPED_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}


def _source_files():
    return [p for p in sorted(PACKAGE_DIR.rglob("*.py")) if "__pycache__" not in p.parts]


def _non_comment_fragments(path: Path):
    """Yield (line_number, text) for every non-comment part of a source file."""
    source = path.read_bytes().decode("utf-8-sig")
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in _SKIPPED_TOKENS:
                continue
            yield token.start[0], token.string
    except (tokenize.TokenError, SyntaxError):
        # Unparseable file: fall back to a line scan that drops whole-line comments.
        for number, line in enumerate(source.splitlines(), start=1):
            if not line.lstrip().startswith("#"):
                yield number, line


def find_hardcoded_model_strings(paths, root=REPO_ROOT):
    """Case-insensitive search for the forbidden patterns outside comments (docstrings count)."""
    hits = []
    for path in paths:
        for number, text in _non_comment_fragments(path):
            lowered = text.lower()
            for pattern in FORBIDDEN:
                if pattern in lowered:
                    hits.append(f"{path.relative_to(root).as_posix()}:{number}: {text.strip()!r}")
    return hits


def test_package_sources_exist():
    assert _source_files(), "impulse_evals/ must contain Python sources"


def test_no_hardcoded_model_strings_in_package():
    hits = find_hardcoded_model_strings(_source_files())
    assert not hits, "Hardcoded model strings in source code:\n" + "\n".join(hits)


def test_scanner_ignores_comments_and_flags_code(tmp_path):
    """Self-check of the scanner: comments are allowed, code and docstrings are not."""
    sample = tmp_path / "impulse_evals" / "sample.py"
    sample.parent.mkdir()
    sample.write_text(
        '"""Docstring naming GPT-x counts as source."""\n'
        "# example model: claude-x in a comment is fine\n"
        "x = 1  # inline comment gpt-x is fine\n"
        'MODEL = "gemini-x"\n',
        encoding="utf-8",
    )
    hits = find_hardcoded_model_strings([sample], root=tmp_path)
    assert len(hits) == 2
    assert ":1:" in hits[0] and "GPT-x" in hits[0]
    assert ":4:" in hits[1] and "gemini-x" in hits[1]

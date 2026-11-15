# SPDX-License-Identifier: Apache-2.0
"""Phase 14: documentation checks (SPEC.md sections 5.1, 17, 21; maintainer decisions in DECISIONS.md).

Written before ``docs/`` exists; every test fails until phase 14 writes the docs.

* Every fenced YAML / TOML block in ``docs/`` that contains the line ``# impulse:validate`` loads:
  YAML parses (and a complete ``config.yaml``, i.e. a mapping with a ``simulator`` key, passes
  ``load_config`` with its system prompt file created as a placeholder); TOML parses and passes
  ``load_toml``. There is at least one such YAML block and one such TOML block.
* Every ``impulse <word>`` in a code block or inline code span of ``docs/`` and ``README.md`` names a
  real subcommand (``impulse <word> --help`` exits 0); ``docs/cli.md`` shows every subcommand.
* Relative links (and ``#anchors`` into Markdown pages) in ``docs/`` and ``README.md`` resolve.
* Maintainer-required texts: the Python sentence in the README install section and in the
  ``docs/run-for-free.md`` prerequisites; "Why not Python 3.10?" in ``docs/faq.md``; the Microsoft
  Visual C++ Redistributable (x64) for a missing ``vcruntime140_1.dll`` in the run-for-free
  troubleshooting step; every page has an SPDX header, an H1 title, a one-sentence intro and a final
  "Next steps" section with links; ``docs/verified-setups.md`` has one table: the header, the
  separator and the TEMPLATE row, then any number of real rows (maintainer decision 2026-10-25,
  superseding "TEMPLATE row only"), each with the header's cell count, a ``YYYY-MM-DD`` date first,
  the marker "# example, check the current name" when its models cell is filled, and every
  acronym of SPEC.md section 21.6 it uses spelled out in the row itself; no model names
  (``claude-``, ``gemini-``, ``gpt-``) outside lines marked "# example, check the current name".
* SPEC.md section 21.6: on every docs page the acronyms WER, TTFA, STT, TTS, DPDP, TRAI, DLT, KYC
  and PHI are spelled out on first use (the expansion appears in the page no later than the
  paragraph of the first use in running text; headings are not uses).
* ``docs/cli.md`` names every flag of SPEC.md section 17; ``docs/adapters.md`` has the ``#pipecat``
  anchor SPEC.md sections 9 and 20 point to; ``examples/pipecat/README.md`` carries the bot-side
  checklist essentials (``-t eval``, the ``/ws-client`` path, "check against your Pipecat version").
"""

import re
import sys
import tomllib
import urllib.parse
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"
LICENSING = REPO_ROOT / "LICENSING.md"

DOC_PAGES = [
    "index.md",
    "getting-started.md",
    "concepts.md",
    "run-for-free.md",
    "configuration.md",
    "cli.md",
    "metrics.md",
    "compliance-packs.md",
    "voice-testing.md",
    "adapters.md",
    "writing-scenarios.md",
    "ci-integration.md",
    "calibration.md",
    "cookbook.md",
    "troubleshooting.md",
    "privacy-and-security.md",
    "architecture.md",
    "contributing.md",
    "faq.md",
    "verified-setups.md",
    "changelog.md",
]
SUBCOMMANDS = ("init", "doctor", "generate", "run", "report", "compare", "convert-failure", "calibrate")
PYTHON_SENTENCE = "Requires Python 3.11 or newer (tested on 3.11, 3.12 and 3.14). Python 3.10 is not supported."
MD_HEADER = "<!-- SPDX-License-Identifier: Apache-2.0 -->"
MODEL_PATTERNS = ("claude-", "gemini-", "gpt-")
MODEL_MARKER = "# example, check the current name"

VALIDATE_MARKER = re.compile(r"^\s*#\s*impulse:validate\s*$", re.MULTILINE)
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
COMMAND = re.compile(r"(?<![\w./\\-])impulse[ \t]+([A-Za-z][\w-]*)")
LINK = re.compile(r"!?\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)\s>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "Dr.", "No.")

# (page, text the page must contain) from SPEC.md sections 9, 21.2, 21.3, 21.5 and KICKOFF.md 5b.
REQUIRED_MENTIONS = [
    ("cli.md", "--fail-on"),
    ("cli.md", "--embed-audio"),
    ("cli.md", "--no-embed-audio"),
    ("cli.md", "impulse compare"),
    ("ci-integration.md", "impulse compare"),
    ("ci-integration.md", "--fail-on"),
    ("metrics.md", "UNIV-023"),
    ("metrics.md", "UNIV-025"),
    ("compliance-packs.md", "Apache License 2.0"),
    ("compliance-packs.md", "hipaa"),
    ("compliance-packs.md", "opt-in"),
    ("compliance-packs.md", "Technical signal only. Not legal certification."),
    ("adapters.md", "-t eval"),
    ("adapters.md", "/ws-client"),
    ("adapters.md", "check against your Pipecat version"),
    ("adapters.md", "get_system_prompt"),
    ("adapters.md", "FULL"),
    ("adapters.md", "Apache"),
    ("adapters.md", "plugin_types.py"),
    ("contributing.md", "Apache"),
    ("contributing.md", "plugin_types.py"),
    ("writing-scenarios.md", "inject_events"),
    ("writing-scenarios.md", "v2"),
    ("writing-scenarios.md", "seed"),
    ("writing-scenarios.md", "max_turns"),
    ("writing-scenarios.md", "expected_tool_calls"),
    ("writing-scenarios.md", "forbidden_tool_calls"),
    ("run-for-free.md", "ollama pull YOUR_FAST_MODEL"),
    ("run-for-free.md", "ollama pull YOUR_PRIMARY_MODEL"),
    ("run-for-free.md", "pip install impulse-evals[local]"),
    ("run-for-free.md", "impulse init --profile free"),
    ("run-for-free.md", "impulse doctor"),
    ("run-for-free.md", "YOUR_TTS_MODEL"),
    ("run-for-free.md", "YOUR_STT_MODEL"),
    ("faq.md", "Why not Python 3.10?"),
]
README_MENTIONS = [
    "formerly PULSE",
    "LICENSING.md",
    "Apache-2.0",
    "NOTICE",
    "Trademark",
    "heard_text",
    "tool_calls",
    "Talk2Neo",
    "Pipecat",
    "config.toml",
]
# (page, text) compared case-insensitively; "README.md" is the repository README.
REQUIRED_MENTIONS_ANY_CASE = [
    ("README.md", "privacy"),
    ("cli.md", "exit code"),
    ("ci-integration.md", "exit code"),
    ("adapters.md", "one client at a time"),
    ("run-for-free.md", "not verified"),
    ("run-for-free.md", "examples/free/config.yaml"),
    ("run-for-free.md", "sample_agent.py"),
    ("run-for-free.md", "ffmpeg"),
]
# Every flag of the SPEC.md section 17 command table.
CLI_FLAGS = (
    "--profile", "--language", "--no-ping", "--non-interactive", "--prompt", "--domain", "--count", "--out",
    "--yes", "--voice-mode", "--reporter", "--dry-run", "--concurrency", "--fail-on", "--embed-audio",
    "--no-embed-audio",
)
# SPEC.md section 21.6: acronym -> its spelled-out form (case-insensitive; a space or hyphen between words).
ACRONYMS = {
    "WER": r"word[\s-]+error[\s-]+rate",
    "TTFA": r"time[\s-]+to[\s-]+first[\s-]+audio",
    "STT": r"speech[\s-]+to[\s-]+text",
    "TTS": r"text[\s-]+to[\s-]+speech",
    "DPDP": r"digital[\s-]+personal[\s-]+data[\s-]+protection",
    "TRAI": r"telecom(?:munications?)?[\s-]+regulatory[\s-]+authority[\s-]+of[\s-]+india",
    "DLT": r"distributed[\s-]+ledger[\s-]+technology",
    "KYC": r"know[\s-]+your[\s-]+customer",
    "PHI": r"protected[\s-]+health[\s-]+information",
}
PIPECAT_EXAMPLE_README = REPO_ROOT / "examples" / "pipecat" / "README.md"
PIPECAT_EXAMPLE_MENTIONS = ["-t eval", "/ws-client", "check against your Pipecat version"]


# ---------------------------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------------------------


def _read(path):
    return Path(path).read_bytes().decode("utf-8-sig")


def _split(text):
    """``(prose, blocks, unterminated)``: prose = [(line_no, line)] outside fenced code;
    blocks = [(info, body, line_no)] of fenced code blocks; unterminated = line_no of an open fence."""
    prose, blocks = [], []
    fence = info = start = None
    body = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence, info, body, start = match.group(1), match.group(2).strip(), [], number
            else:
                prose.append((number, line))
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) and not match.group(2).strip():
            blocks.append((info, "\n".join(body), start))
            fence = None
        else:
            body.append(line)
    return prose, blocks, (start if fence is not None else None)


def _headings(prose):
    found = []
    for number, line in prose:
        match = HEADING.match(line)
        if match:
            found.append((number, len(match.group(1)), match.group(2)))
    return found


def _slug(title):
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
    title = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return title.replace(" ", "-")


def _anchors(path):
    prose, _blocks, _open = _split(_read(path))
    anchors, seen = set(), {}
    for _number, _level, title in _headings(prose):
        slug = _slug(title)
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    for _number, line in prose:
        anchors.update(m.lower() for m in re.findall(r"<a\s+[^>]*(?:id|name)=[\"']([^\"']+)[\"']", line))
    return anchors


def _links(path):
    """Link targets outside code (inline links, images, reference definitions)."""
    prose, _blocks, _open = _split(_read(path))
    targets = []
    for number, line in prose:
        line = INLINE_CODE.sub("", line)
        targets += [(number, target) for target in LINK.findall(line)]
        ref = REF_DEF.match(line)
        if ref:
            targets.append((number, ref.group(1)))
    return targets


def _resolve(source, target):
    """``(path, fragment)`` for a relative link, or None for an external one."""
    if target.startswith("//") or SCHEME.match(target):
        return None
    path_part, _sep, fragment = target.partition("#")
    path_part = urllib.parse.unquote(path_part)
    path = Path(source) if not path_part else (Path(source).parent / path_part)
    return path.resolve(), urllib.parse.unquote(fragment)


def _commands_in(path):
    prose, blocks, _open = _split(_read(path))
    found = []
    for _info, body, start in blocks:
        found += [(start, m.group(1)) for m in COMMAND.finditer(body)]
    for number, line in prose:
        for span in INLINE_CODE.findall(line):
            found += [(number, m.group(1)) for m in COMMAND.finditer(span)]
    return found


def _doc_files():
    assert DOCS_DIR.is_dir(), "docs/ does not exist yet (phase 14)"
    missing = [page for page in DOC_PAGES if not (DOCS_DIR / page).is_file()]
    assert not missing, f"docs pages missing (SPEC.md section 21.2): {missing}"
    return sorted(path for path in DOCS_DIR.rglob("*.md"))


def _all_markdown():
    return _doc_files() + [README]


def _rel(path):
    return Path(path).relative_to(REPO_ROOT).as_posix()


def _sections(text, title_pattern):
    """Text of every section whose heading matches ``title_pattern`` (each up to the next heading of
    the same or a higher level)."""
    prose, _blocks, _open = _split(text)
    headings = _headings(prose)
    lines = text.splitlines()
    found = []
    for index, (number, level, title) in enumerate(headings):
        if re.search(title_pattern, title, re.IGNORECASE):
            end = len(lines)
            for later_number, later_level, _title in headings[index + 1:]:
                if later_level <= level:
                    end = later_number - 1
                    break
            found.append("\n".join(lines[number:end]))
    return found


STEP_LINE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|\d+[.)]\s+|[-*+]\s+)")


def _step_start(text, word):
    """Character offset of the first heading or list item (outside code) whose line contains ``word``
    (case-insensitive), or -1. A link in a "Next steps" list naming the word also counts, so callers
    look for the FIRST such line."""
    prose_numbers = {number for number, _line in _split(text)[0]}
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if number in prose_numbers and STEP_LINE.match(line) and word in line.lower():
            return offset
        offset += len(line)
    return -1


# ---------------------------------------------------------------------------------------------
# Docs tree, headers, and page structure (SPEC.md sections 5.1, 21.2, 21.6)
# ---------------------------------------------------------------------------------------------


def test_docs_tree_is_complete():
    _doc_files()
    assert README.is_file(), "README.md is missing"


@pytest.mark.parametrize("page", DOC_PAGES)
def test_page_starts_with_the_spdx_header(page):
    first = _read(DOCS_DIR / page).splitlines()[0].strip()
    assert first == MD_HEADER


def test_readme_and_licensing_start_with_the_spdx_header():
    for path in (README, LICENSING):
        assert _read(path).splitlines()[0].strip() == MD_HEADER, _rel(path)


def _intro(text):
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].strip().startswith("<!--")):
        index += 1
    assert index < len(lines) and re.match(r"^# \S", lines[index]), "the page starts with an H1 title"
    index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    paragraph = []
    while index < len(lines) and lines[index].strip():
        paragraph.append(lines[index].strip())
        index += 1
    return " ".join(paragraph)


@pytest.mark.parametrize("page", DOC_PAGES)
def test_page_has_a_one_sentence_intro(page):
    intro = _intro(_read(DOCS_DIR / page))
    assert intro, "a one-sentence intro follows the title"
    assert not re.match(r"^(#|[-*+] |\d+[.)] |\||>|```|~~~|<)", intro), f"the intro must be prose: {intro!r}"
    assert re.search(r"[.!?][\"')\]]*$", intro), f"the intro must be a full sentence: {intro!r}"
    masked = intro
    for abbreviation in ABBREVIATIONS:
        masked = masked.replace(abbreviation, abbreviation.replace(".", ""))
    sentences = re.split(r"(?<=[.!?])\s+(?=[\"'(\[`]?[A-Z0-9])", masked)
    assert len(sentences) == 1, f"the intro must be ONE sentence: {intro!r}"


@pytest.mark.parametrize("page", DOC_PAGES)
def test_page_ends_with_next_steps_links(page):
    text = _read(DOCS_DIR / page)
    prose, _blocks, _open = _split(text)
    headings = _headings(prose)
    assert headings, "the page has headings"
    number, _level, title = headings[-1]
    assert title.strip().rstrip(":").lower() == "next steps", f"last heading is {title!r}, not 'Next steps'"
    section = "\n".join(line for line_number, line in prose if line_number > number)
    assert LINK.search(section), "the Next steps section links to other pages"


def test_no_unterminated_code_fences():
    open_fences = []
    for path in _all_markdown():
        _prose, _blocks, open_at = _split(_read(path))
        if open_at is not None:
            open_fences.append(f"{_rel(path)}:{open_at}")
    assert not open_fences, f"unterminated code fences: {open_fences}"


# ---------------------------------------------------------------------------------------------
# # impulse:validate blocks
# ---------------------------------------------------------------------------------------------


def _validate_blocks():
    found = []
    for path in _doc_files():
        _prose, blocks, _open = _split(_read(path))
        for info, body, start in blocks:
            if VALIDATE_MARKER.search(body):
                language = info.split()[0].lower() if info.split() else ""
                found.append((path, start, language, body))
    return found


def _load_full_config(body, data, folder):
    from impulse_evals.config import load_config

    folder.mkdir(parents=True, exist_ok=True)
    prompt = (data.get("agent") or {}).get("system_prompt")
    if isinstance(prompt, str) and prompt.strip() and not Path(prompt).is_absolute():
        target = folder / prompt
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("You are a placeholder agent used to validate the docs.", encoding="utf-8")
    config_path = folder / "config.yaml"
    config_path.write_text(body, encoding="utf-8")
    load_config(config_path)


def test_validate_blocks_load(tmp_path):
    from impulse_evals.config import load_toml

    blocks = _validate_blocks()
    languages = {language for _path, _start, language, _body in blocks}
    assert languages & {"yaml", "yml"}, "at least one YAML block is marked # impulse:validate"
    assert "toml" in languages, "at least one TOML block is marked # impulse:validate"
    errors = []
    for index, (path, start, language, body) in enumerate(blocks):
        where = f"{_rel(path)}:{start}"
        folder = tmp_path / f"block_{index}" / "project"
        try:
            if language in ("yaml", "yml"):
                data = yaml.safe_load(body)
                assert data is not None, "empty YAML block"
                if isinstance(data, dict) and "simulator" in data:
                    _load_full_config(body, data, folder)
            elif language == "toml":
                tomllib.loads(body)
                folder.mkdir(parents=True, exist_ok=True)
                toml_path = folder / "config.toml"
                toml_path.write_text(body, encoding="utf-8")
                load_toml(toml_path)
            else:
                raise AssertionError(f"# impulse:validate is only for yaml / toml blocks, not {language!r}")
        except Exception as exc:  # collect every failing block
            errors.append(f"{where} ({language}): {type(exc).__name__}: {exc}")
    assert not errors, "blocks marked # impulse:validate do not load:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------------------------
# impulse commands
# ---------------------------------------------------------------------------------------------


def test_every_impulse_command_names_a_real_subcommand(cli):
    used = {}
    for path in _all_markdown():
        for number, name in _commands_in(path):
            used.setdefault(name, f"{_rel(path)}:{number}")
    assert used, "the docs show impulse commands"
    unknown = []
    for name, where in sorted(used.items()):
        result = cli.invoke(name, "--help")
        if result.exit_code != 0:
            unknown.append(f"impulse {name} ({where})")
    assert not unknown, f"not real subcommands: {unknown}"


def test_cli_page_shows_every_subcommand():
    shown = {name for _number, name in _commands_in(DOCS_DIR / "cli.md")}
    missing = [name for name in SUBCOMMANDS if name not in shown]
    assert not missing, f"docs/cli.md does not show: {missing}"


# ---------------------------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------------------------


def test_relative_links_resolve():
    broken = []
    for path in _all_markdown():
        for number, target in _links(path):
            resolved = _resolve(path, target)
            if resolved is None:
                continue
            target_path, fragment = resolved
            where = f"{_rel(path)}:{number} -> {target}"
            if not target_path.exists():
                broken.append(f"{where} (missing file)")
            elif fragment and target_path.suffix.lower() == ".md" and fragment.lower() not in _anchors(target_path):
                broken.append(f"{where} (missing anchor)")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_readme_links_every_docs_page_and_licensing():
    _doc_files()
    linked = set()
    for _number, target in _links(README):
        resolved = _resolve(README, target)
        if resolved is not None:
            linked.add(resolved[0])
    expected = {(DOCS_DIR / page).resolve() for page in DOC_PAGES} | {LICENSING.resolve()}
    missing = sorted(_rel(path) for path in expected - linked)
    assert not missing, f"README.md does not link: {missing}"


# ---------------------------------------------------------------------------------------------
# Maintainer-required texts
# ---------------------------------------------------------------------------------------------


def test_readme_install_section_has_the_python_sentence():
    sections = _sections(_read(README), r"\binstall")
    assert sections, "README.md has an install section (a heading containing 'install')"
    assert any(PYTHON_SENTENCE in section for section in sections), "the sentence is in an install section"


def test_run_for_free_prerequisites_have_the_python_sentence():
    text = _read(DOCS_DIR / "run-for-free.md")
    assert PYTHON_SENTENCE in text
    prerequisites = _step_start(text, "prerequisites")
    sentence = text.find(PYTHON_SENTENCE)
    ollama_pull = text.lower().find("ollama pull")
    assert prerequisites >= 0, "docs/run-for-free.md has a Prerequisites step (heading or numbered item)"
    assert prerequisites <= sentence, "the sentence is in the prerequisites step"
    assert ollama_pull == -1 or sentence < ollama_pull, "the prerequisites come before step 2 (Ollama)"


def test_run_for_free_troubleshooting_mentions_the_vc_redistributable():
    text = _read(DOCS_DIR / "run-for-free.md")
    start = _step_start(text, "troubleshooting")
    assert start >= 0, "docs/run-for-free.md has a Troubleshooting step (heading or numbered item)"
    tail = text[start:]
    for needed in ("vcruntime140_1.dll", "Microsoft Visual C++ Redistributable", "x64"):
        assert needed in tail, f"troubleshooting does not mention {needed}"


def test_faq_answers_why_not_python_310():
    text = _read(DOCS_DIR / "faq.md")
    position = text.find("Why not Python 3.10?")
    assert position >= 0
    after = text[position + len("Why not Python 3.10?"):]
    lines = after.splitlines()
    answer_lines = [lines[0]] if lines and lines[0].strip(" *_#:") else []
    started = bool(answer_lines)
    for line in lines[1:]:
        if not line.strip():
            if started:
                break
            continue
        if HEADING.match(line):
            break
        answer_lines.append(line)
        started = True
    answer = " ".join(answer_lines).lower()
    assert "pipecat" in answer, f"the answer names Pipecat's eval mode: {answer!r}"
    assert "october 2026" in answer, f"the answer gives the Python 3.10 end of life: {answer!r}"


def _verified_setups_table():
    """The rows (stripped lines) of the one table in ``docs/verified-setups.md``."""
    prose, _blocks, _open = _split(_read(DOCS_DIR / "verified-setups.md"))
    tables, current, previous = [], [], None
    for number, line in prose:
        if line.strip().startswith("|"):
            if previous is not None and number != previous + 1 and current:
                tables.append(current)
                current = []
            current.append(line.strip())
            previous = number
        elif current:
            tables.append(current)
            current, previous = [], None
    if current:
        tables.append(current)
    assert len(tables) == 1, f"expected exactly one table, found {len(tables)}"
    return tables[0]


def _cells(row):
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def test_verified_setups_table_starts_with_header_separator_and_template_row():
    """Superseded test (maintainer decision 2026-10-25): ``..._ships_with_header_and_template_row_only``
    required exactly three rows; real rows are now allowed after the TEMPLATE row."""
    rows = _verified_setups_table()
    assert len(rows) >= 3, f"header, separator and the TEMPLATE row at least; found {len(rows)} rows"
    assert re.match(r"^\|?\s*:?-{3,}", rows[1]), "the second row is the header separator"
    assert "TEMPLATE" in rows[2]
    assert not any("TEMPLATE" in row for row in rows[3:]), "only one TEMPLATE row"


def test_verified_setups_real_rows_follow_the_rules():
    """Every row after the TEMPLATE row: the header's cell count, a ``YYYY-MM-DD`` date first, the model
    marker when the models cell is filled, and each section 21.6 acronym it uses spelled out in the row."""
    rows = _verified_setups_table()
    header = [cell.lower() for cell in _cells(rows[0])]
    assert "models" in header, "the header has a models column"
    models_at = header.index("models")
    problems = []
    for number, row in enumerate(rows[3:], start=1):
        cells = _cells(row)
        if len(cells) != len(header):
            problems.append(f"row {number}: {len(cells)} cells, the header has {len(header)}")
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
            problems.append(f"row {number}: the first cell is not a YYYY-MM-DD date: {cells[0]!r}")
        if cells[models_at] and MODEL_MARKER not in row:
            problems.append(f"row {number}: model names without {MODEL_MARKER!r}")
        visible = re.sub(r"\]\([^)]*\)", "]", INLINE_CODE.sub("", row))
        for acronym, expansion in ACRONYMS.items():
            if re.search(rf"(?<![\w-]){acronym}(?![\w-])", visible) and not re.search(expansion, row, re.IGNORECASE):
                problems.append(f"row {number}: {acronym} is not spelled out in the row")
    assert not problems, "docs/verified-setups.md rows:\n" + "\n".join(problems)


def test_verified_setups_row_rules_catch_a_bad_row(monkeypatch):
    """The row check above fails a row with an unmarked model name and an unexpanded acronym."""
    template = _verified_setups_table()
    bad = "| 2026-10-24 | 0.1.0 | OS | 3.12 | CPU | LiveKit, TTS to a proxy | judge `gemini-x` | run | ok | name |"
    monkeypatch.setattr(sys.modules[__name__], "_verified_setups_table", lambda: [*template[:3], bad])
    with pytest.raises(AssertionError) as caught:
        test_verified_setups_real_rows_follow_the_rules()
    message = str(caught.value)
    assert MODEL_MARKER in message and "TTS is not spelled out" in message


def test_no_model_names_outside_marked_lines():
    hits = []
    for path in _all_markdown():
        for number, line in enumerate(_read(path).splitlines(), start=1):
            lowered = line.lower()
            if any(pattern in lowered for pattern in MODEL_PATTERNS) and MODEL_MARKER not in line:
                hits.append(f"{_rel(path)}:{number}: {line.strip()}")
    assert not hits, "model names without '# example, check the current name':\n" + "\n".join(hits)


@pytest.mark.parametrize("page, needed", REQUIRED_MENTIONS, ids=[f"{p}:{n}" for p, n in REQUIRED_MENTIONS])
def test_page_mentions(page, needed):
    assert needed in _read(DOCS_DIR / page), f"docs/{page} must mention {needed!r}"


@pytest.mark.parametrize("needed", README_MENTIONS)
def test_readme_mentions(needed):
    assert needed in _read(README), f"README.md must mention {needed!r}"


@pytest.mark.parametrize(
    "page, needed", REQUIRED_MENTIONS_ANY_CASE, ids=[f"{p}:{n}" for p, n in REQUIRED_MENTIONS_ANY_CASE]
)
def test_page_mentions_any_case(page, needed):
    path = README if page == "README.md" else DOCS_DIR / page
    assert needed.lower() in _read(path).lower(), f"{_rel(path)} must mention {needed!r}"


@pytest.mark.parametrize("flag", CLI_FLAGS)
def test_cli_page_documents_every_flag(flag):
    text = _read(DOCS_DIR / "cli.md")
    assert re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text), f"docs/cli.md does not document {flag}"


def test_adapters_page_has_the_pipecat_anchor():
    """SPEC.md sections 9 and 20 point to ``docs/adapters.md#pipecat``."""
    assert "pipecat" in _anchors(DOCS_DIR / "adapters.md")


@pytest.mark.parametrize("needed", PIPECAT_EXAMPLE_MENTIONS)
def test_pipecat_example_readme_mentions(needed):
    """SPEC.md section 9: the bot-side checklist is documented in examples/pipecat/README.md too."""
    assert needed in _read(PIPECAT_EXAMPLE_README), f"examples/pipecat/README.md must mention {needed!r}"


def _first_use(prose, acronym):
    """``(index into prose, line)`` of the first running-text use of ``acronym`` (headings, inline code,
    link targets, and HTML comments are not uses), or None."""
    pattern = re.compile(rf"(?<![\w-]){acronym}(?![\w-])")
    for index, (_number, line) in enumerate(prose):
        if HEADING.match(line):
            continue
        visible = re.sub(r"<!--.*?-->", "", INLINE_CODE.sub("", line))
        visible = re.sub(r"\]\([^)]*\)", "]", visible)
        if pattern.search(visible):
            return index
    return None


@pytest.mark.parametrize("page", DOC_PAGES)
def test_acronyms_are_spelled_out_on_first_use(page):
    prose, _blocks, _open = _split(_read(DOCS_DIR / page))
    problems = []
    for acronym, expansion in ACRONYMS.items():
        index = _first_use(prose, acronym)
        if index is None:
            continue
        end = index
        while end + 1 < len(prose) and prose[end + 1][1].strip() and prose[end + 1][0] == prose[end][0] + 1:
            end += 1  # the rest of the paragraph (consecutive non-blank lines)
        allowed = "\n".join(line for _number, line in prose[: end + 1])
        if not re.search(expansion, allowed, re.IGNORECASE):
            problems.append(f"{acronym} (first used on line {prose[index][0]})")
    assert not problems, f"docs/{page}: acronyms not spelled out on first use: {problems}"


# ---------------------------------------------------------------------------------------------
# Audit batch 2 (AUDIT.md finding 9): docs that show dry-run console output show what the CLI
# prints: `impulse run --dry-run` lists the dry-run run warning under "Warnings:", and
# `impulse compare` of dry runs prints one "Note: run {run_id} is a dry run: ..." line per dry run
# (run A first) after the table.
# ---------------------------------------------------------------------------------------------

DRY_RUN_WARNING = "Dry run: judge metrics and assertions were skipped; results are not verdicts."
DRY_RUN_NOTE = "Note: run {run_id} is a dry run: judge metrics and assertions were skipped, so its results are not verdicts."
DRY_RUN_SUMMARY = re.compile(r"^Run (\S+) \(dry run\): ")
COMPARE_LINE = re.compile(r"^Compare (\S+) \([^)]*\) -> (\S+) \([^)]*\): ")
RUN_ID_RE = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
RUN_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
# Pages whose dry-run blocks show `impulse run examples/config.yaml --dry-run` and the compare of two
# such dry runs (verified outputs, per the pages' own text).
EXAMPLES_DRY_RUN_PAGES = ("cli.md", "getting-started.md")
EXAMPLES_DRY_COMPARE_PAGES = ("cli.md", "ci-integration.md")


def _blocks_with_prose(path):
    """``(block body, the prose paragraph right before the block)`` for every fenced block."""
    text = _read(path)
    lines = text.splitlines()
    _prose, blocks, _open = _split(text)
    found = []
    for _info, body, start in blocks:
        before, index = [], start - 2   # start is the 1-based fence line
        while index >= 0 and not lines[index].strip():
            index -= 1
        while index >= 0 and lines[index].strip() and not FENCE.match(lines[index]):
            before.insert(0, lines[index])
            index -= 1
        found.append((body, "\n".join(before)))
    return found


def _dry_run_blocks():
    """``(path, body)`` of every fenced block (docs and README) showing an `impulse run --dry-run` summary."""
    return [
        (path, body)
        for path in _all_markdown()
        for body, _before in _blocks_with_prose(path)
        if any(DRY_RUN_SUMMARY.match(line.strip()) for line in body.splitlines())
    ]


def _dry_compare_blocks():
    """``(path, body)`` of every fenced `impulse compare` output block introduced as a compare of dry runs."""
    return [
        (path, body)
        for path in _all_markdown()
        for body, before in _blocks_with_prose(path)
        if body.strip() and COMPARE_LINE.match(body.strip().splitlines()[0]) and "dry run" in before.lower()
    ]


def _normalised(text):
    """Console text with run IDs, timestamps, and path separators made comparable, whitespace collapsed."""
    text = RUN_ID_RE.sub("<run_id>", text)
    text = RUN_TIMESTAMP_RE.sub("<timestamp>", text)
    return " ".join(text.replace("\\", "/").split())


def test_docs_dry_run_blocks_show_the_dry_run_warning():
    blocks = _dry_run_blocks()
    pages = sorted({Path(path).name for path, _body in blocks})
    assert set(EXAMPLES_DRY_RUN_PAGES) <= set(pages), pages
    for path, body in blocks:
        lines = [line.rstrip() for line in body.splitlines()]
        assert "Warnings:" in lines, f"{_rel(path)}: the dry-run output block has no Warnings: line"
        assert f"  - {DRY_RUN_WARNING}" in lines, f"{_rel(path)}: the dry-run output block does not show the dry-run warning"


def test_docs_dry_run_compare_blocks_show_a_note_per_dry_run():
    blocks = _dry_compare_blocks()
    pages = sorted({Path(path).name for path, _body in blocks})
    assert set(EXAMPLES_DRY_COMPARE_PAGES) <= set(pages), pages
    for path, body in blocks:
        lines = [line.strip() for line in body.splitlines()]
        run_a, run_b = COMPARE_LINE.match(lines[0]).groups()
        notes = [line for line in lines if line.startswith("Note: run ")]
        expected = [DRY_RUN_NOTE.format(run_id=run_a), DRY_RUN_NOTE.format(run_id=run_b)]
        assert notes == expected, f"{_rel(path)}: the compare of two dry runs must show {expected}, not {notes}"


def _examples_dry_run(cli, project):
    before = set(project.run_dirs())
    result = cli.invoke("run", REPO_ROOT / "examples" / "config.yaml", "--dry-run")
    assert result.exit_code == 0, cli.text(result)
    (run_dir,) = set(project.run_dirs()) - before
    return result, project.read_run(run_dir)


def test_docs_dry_run_blocks_match_what_the_cli_prints(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): the verified dry-run output of examples/config.yaml
    # in the docs matches the CLI line for line, up to run IDs, timestamps, and path separators.
    result, _run = _examples_dry_run(cli, impulse_project)
    assert mock_llm.call_count == 0
    printed = _normalised(cli.text(result))
    by_page = {Path(path).name: body for path, body in _dry_run_blocks()}
    for page in EXAMPLES_DRY_RUN_PAGES:
        assert page in by_page, f"docs/{page} has no dry-run output block"
        assert _normalised(by_page[page]) == printed, f"docs/{page}: the dry-run block differs from the CLI output"


def test_docs_dry_run_compare_blocks_match_what_the_cli_prints(impulse_project, mock_llm, cli):
    # Regression guard (passes before the fix too): the verified compare of two dry runs of
    # examples/config.yaml in the docs matches the CLI, up to run IDs, timestamps, and separators.
    _first, run_a = _examples_dry_run(cli, impulse_project)
    _second, run_b = _examples_dry_run(cli, impulse_project)
    result = cli.invoke("compare", run_a.run_id, run_b.run_id)
    assert result.exit_code == 0, cli.text(result)
    assert mock_llm.call_count == 0
    printed = _normalised(cli.text(result))
    by_page = {Path(path).name: body for path, body in _dry_compare_blocks()}
    for page in EXAMPLES_DRY_COMPARE_PAGES:
        assert page in by_page, f"docs/{page} has no compare-of-dry-runs output block"
        assert _normalised(by_page[page]) == printed, f"docs/{page}: the compare block differs from the CLI output"


# ---------------------------------------------------------------------------------------------
# Audit batch 2 (AUDIT.md finding 23; SPEC.md section 21.1 rule 2): the CI sentence in
# docs/contributing.md is labelled "not verified".
# ---------------------------------------------------------------------------------------------

CI_SENTENCE = (
    "The workflow in `.github/workflows/ci.yml` runs the same command on Python 3.11, 3.12, and 3.14 "
    "(not verified: CI has not run yet)."
)
OLD_CI_CLAIM = "CI runs the same command on Python 3.11, 3.12, and 3.14"


def test_contributing_ci_sentence_is_labelled_not_verified():
    text = " ".join(_read(DOCS_DIR / "contributing.md").split())
    labelled = CI_SENTENCE in text
    unlabelled = OLD_CI_CLAIM in text
    assert labelled, "docs/contributing.md must carry the labelled CI sentence"
    assert not unlabelled, "the unlabelled CI claim is still in docs/contributing.md"


# ---------------------------------------------------------------------------------------------
# Audit batch 3 (AUDIT.md finding 48; SPEC.md section 21.6): README.md spells out DPDP, TRAI, DLT
# and KYC, and examples/pipecat/README.md spells out TTS, STT and WER, on first use (same rule as
# the docs pages: the expansion appears no later than the paragraph of the first running-text use);
# docs/index.md names the init command that was actually run in its "verified" list.
# ---------------------------------------------------------------------------------------------

README_ACRONYMS = ("DPDP", "TRAI", "DLT", "KYC")
PIPECAT_README_ACRONYMS = ("TTS", "STT", "WER")
VERIFIED_INIT_COMMAND = "impulse init --profile free --non-interactive --language en-IN"


def _acronym_spelled_out_on_first_use(path, acronym):
    """True when ``acronym`` is not used in running text of ``path``, or its expansion (``ACRONYMS``)
    appears in the page no later than the paragraph of its first running-text use."""
    prose, _blocks, _open = _split(_read(path))
    index = _first_use(prose, acronym)
    if index is None:
        return True
    end = index
    while end + 1 < len(prose) and prose[end + 1][1].strip() and prose[end + 1][0] == prose[end][0] + 1:
        end += 1   # the rest of the paragraph (consecutive non-blank lines)
    allowed = "\n".join(line for _number, line in prose[: end + 1])
    return re.search(ACRONYMS[acronym], allowed, re.IGNORECASE) is not None


@pytest.mark.parametrize("acronym", README_ACRONYMS)
def test_readme_spells_out_compliance_acronyms_on_first_use(acronym):
    assert _acronym_spelled_out_on_first_use(README, acronym), (
        f"README.md: {acronym} is not spelled out on first use"
    )


@pytest.mark.parametrize("acronym", PIPECAT_README_ACRONYMS)
def test_pipecat_example_readme_spells_out_voice_acronyms_on_first_use(acronym):
    assert _acronym_spelled_out_on_first_use(PIPECAT_EXAMPLE_README, acronym), (
        f"examples/pipecat/README.md: {acronym} is not spelled out on first use"
    )


def test_index_verified_list_names_the_exact_init_command():
    sections = _sections(_read(DOCS_DIR / "index.md"), r"verified")
    assert sections, "docs/index.md has no section about what was verified"
    spans = [span for section in sections for span in INLINE_CODE.findall(section)]
    init_spans = [span.strip() for span in spans if span.strip().startswith("impulse init")]
    assert VERIFIED_INIT_COMMAND in init_spans, f"docs/index.md must name `{VERIFIED_INIT_COMMAND}`, not {init_spans}"
    bare = [span for span in init_spans if "--non-interactive" in span and "--language" not in span]
    assert not bare, f"docs/index.md lists an init command that exits 2 (no --language): {bare}"

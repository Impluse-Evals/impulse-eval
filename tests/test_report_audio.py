# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 2, Part B: report audio (FIXES-ROUND-2-SPEC.md B1-B4; reporter.render_html, SPEC.md
section 16). B5 (compressed audio) is not implemented, so it has no test.

* B1: at the top of each scenario's body (before the inject note and the error box), when
  ``transcript.recording_path`` is set: ``<div class="recording">Full call recording (left channel:
  caller, right channel: agent)</div>`` followed by the player for ``recording_path`` (same embed /
  link / refusal rules as turn audio).
* B2: every turn with ``audio_path`` gets a player, user turns as well as agent turns (after the
  turn's text lines).
* B3 (revised 2026-10-25, maintainer: report size): in the Assertions and Metrics tables, the turn
  cell of a failed assertion or a poor (counted) metric whose ``turn_index`` names a turn with
  ``audio_path`` shows, after the turn link, an in-page link ``Hear turn N`` to that turn's player
  (``href="#scenario-S-turn-N-audio"``), never a second player. The turn's player in the transcript
  carries that id (on the element that holds it). No script. Passed, skipped, warning and
  informational rows get neither.
* B4: for each scenario with ``voice_mode`` true and at least one turn with ``end_ms``, one static
  inline ``<svg class="timeline">`` before the transcript list: caller and agent rows, one rect per
  turn with ``end_ms`` (``timestamp_ms`` to ``end_ms``), a ``dead-air`` rect between a user turn's
  ``end_ms`` and the next agent turn's ``timestamp_ms`` when that agent turn's ``latency_ms`` >
  ``dead_air_ms``, class ``slow`` on an agent turn whose ``latency_ms`` > ``ttfa_fail_ms``; each rect
  has a ``<title>``. Thresholds from ``transcript.metadata["impulse_voice"]["thresholds"]`` (the runner
  stores them from ``metrics.voice``), else the SPEC 8.2 defaults (1500, 1500, 3000).
* Size (B-rev, review R3; supersedes "every distinct embedded file"; revised 2026-10-25): the
  embedded-size warning counts the bytes of every embedded player copy in the page (base64 not
  included), and only real copies: a "Hear turn N" link embeds nothing, so a turn named by failed
  checks counts once. The report stays self-contained, escaped, script-free, with no external URL.
* B-rev: timeline values that are not finite numbers or whose magnitude exceeds 10^12 ms are
  treated as missing (that turn is left out of the timeline); a corrupted run.json never stops
  ``render_html``.

Superseded (B-rev): ``test_size_warning_counts_a_shared_file_once`` expected no warning for two
players of one 30 MiB file; it is now ``..._once_per_player`` and expects the 60 MiB warning.

Superseded (2026-10-25, maintainer: report size): ``test_failed_assertion_row_shows_the_turn_player_after_the_turn_link``,
``test_poor_metric_row_shows_the_turn_player_after_the_turn_link`` and
``test_size_warning_counts_every_player_copy_next_to_failed_checks`` expected a second embedded player
next to a failed check; they are now the ``..._links_to_the_turn_player`` tests and
``test_failed_checks_add_no_embedded_copy`` / ``test_size_warning_counts_only_real_embedded_copies``.

Run results are hand-built (``run_builder``) with tiny WAV files written with the stdlib ``wave``
module under ``<tmp>/audio/``.
"""

import base64
import html as html_module
import io
import math
import re
import struct
import wave

import pytest

RECORDING_DIV = '<div class="recording">Full call recording (left channel: caller, right channel: agent)</div>'
DATA_URI = "data:audio/wav;base64,"
SIZE_WARNING = re.compile(
    r"Embedded audio is (\d+(?:\.\d+)?) MB; the HTML report will be large\. "
    r"Use --no-embed-audio to link the files instead\."
)
EXTERNAL_ATTR = re.compile(r"""\b(?:src|href)\s*=\s*["']?\s*(?:https?:|//)""", re.IGNORECASE)
EXTERNAL_CSS = re.compile(r"""url\(\s*["']?\s*(?:https?:|//)""", re.IGNORECASE)
TIMELINE_SVG = re.compile(r"""<svg\b[^>]*\bclass\s*=\s*["'][^"']*\btimeline\b[^"']*["'][^>]*>.*?</svg>""", re.DOTALL)
TITLED_RECT = re.compile(r"<rect\b([^>]*?)>\s*<title>(.*?)</title>\s*</rect>", re.DOTALL)
CLASS_ATTR = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""")
INJECT_NOTE = "inject_events defined but execution reserved for v2; this scenario ran without injection."
SCRIPT = "<script>alert(9)</script>"
DASH = "–"   # the en dash of "start–end ms"

SCENARIO = "booking"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _tone_wav(frequency_hz, seconds=0.05, sample_rate=8000):
    count = int(seconds * sample_rate)
    frames = b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))) for i in range(count)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def _write_large_wav(path, size):
    """A valid 16 kHz mono WAV header whose data chunk is zero-filled up to ``size`` bytes."""
    data_len = size - 44
    header = (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", data_len)
    )
    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(size)


def _render(run, run_dir, embed_audio):
    from impulse_evals.reporter import render_html

    return render_html(run, embed_audio=embed_audio, run_dir=run_dir)


# The voice call (times in ms). Turn 5 has end_ms but no audio file.
#   turn  speaker  start   end    latency  audio
#   1     user     0       1800   -        t1_user
#   2     agent    2400    4000   600      t2_agent
#   3     user     4000    6000   -        t3_user   (text carries a <script> payload)
#   4     agent    8000    9500   2000     t4_agent  (> dead_air 1500: dead-air gap 6000-8000; < 3000: not slow)
#   5     user     9500    10500  -        (none)
#   6     agent    14000   15000  3500     t6_agent  (dead-air gap 10500-14000 and slow)
TURNS = [
    ("user", "I need to book a slot on Tuesday.", 0, 1800, None, "t1_user"),
    ("agent", "Sure, which time suits you?", 2400, 4000, 600.0, "t2_agent"),
    ("user", f"Ten in the morning {SCRIPT}", 4000, 6000, None, "t3_user"),
    ("agent", "Booked for ten.", 8000, 9500, 2000.0, "t4_agent"),
    ("user", "Thanks.", 9500, 10500, None, None),
    ("agent", "You're welcome.", 14000, 15000, 3500.0, "t6_agent"),
]
FREQUENCIES = {"t1_user": 300, "t2_agent": 400, "t3_user": 500, "t4_agent": 600, "t6_agent": 700, "recording": 800}


def _audio_path(stem, key):
    return f"audio/{stem}_{key}.wav"


def _write_audio(run_dir, stem=SCENARIO):
    """Write the WAV files of ``TURNS`` and the recording; returns {key: bytes}."""
    folder = run_dir / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for key, frequency in FREQUENCIES.items():
        data = _tone_wav(frequency)
        (folder / f"{stem}_{key}.wav").write_bytes(data)
        written[key] = data
    return written


def _voice_transcript(b, stem=SCENARIO, *, metadata=None, recording=True, name=None):
    turns = []
    for speaker, text, start, end, latency, key in TURNS:
        fields = {"end_ms": end}
        if latency is not None:
            fields["latency_ms"] = latency
        if key is not None:
            fields["audio_path"] = _audio_path(stem, key)
        turns.append(b.turn(speaker, text, start, **fields))
    return b.transcript(
        name or stem,
        turns,
        voice_mode=True,
        duration_ms=15000,
        metadata=metadata if metadata is not None else {},
        recording_path=_audio_path(stem, "recording") if recording else None,
    )


def _main_result(b, *, metadata=None, name=SCENARIO, **fields):
    return b.scenario_result(
        name,
        transcript=_voice_transcript(b, metadata=metadata, name=name),
        assertions=[
            ("dpdp.consent_before_pii", "fail", {"turn_index": 3}),
            ("dpdp.purpose_disclosure", "pass", {"turn_index": 4}),
            ("dpdp.withdrawal_path", "skip", {"turn_index": 2}),
            ("rbi_kyc.no_credential_request", "fail", {"turn_index": 5, "severity": "high"}),   # turn without audio
        ],
        metrics=[
            ("UNIV-001", 0.0, {"turn_index": 4}),                                   # poor, counted
            ("UNIV-013", 0.5, {"turn_index": 2}),                                   # warning
            ("UNIV-003", 1.0, {"turn_index": 6}),                                   # good
            ("VOICE-011", 0.0, {"turn_index": 1, "informational": True}),           # poor but informational
            ("UNIV-023", 0.0, {"skipped": True, "turn_index": 6}),                  # skipped
        ],
        **fields,
    )


def _voice_run(b, run_dir, *, metadata=None, name=SCENARIO, extra=()):
    written = _write_audio(run_dir)
    run = b.run([_main_result(b, metadata=metadata, name=name), *extra], voice_mode=True)
    return run, written


def _scenario_segment(html_text, index=1):
    """The ``<details>`` of scenario ``index`` (its body up to ``</details>``)."""
    start = html_text.index(f'id="scenario-{index}"')
    end = html_text.index("</details>", start)
    return html_text[start:end]


def _turn_segment(html_text, number, scenario_index=1):
    start = html_text.index(f'id="scenario-{scenario_index}-turn-{number}"')
    end = html_text.index("</li>", start)
    return html_text[start:end]


def _rows(segment):
    return re.findall(r"<tr\b.*?</tr>", segment, re.DOTALL)


def _row_for(segment, check_id):
    rows = [row for row in _rows(segment) if f">{check_id}<" in row]
    assert len(rows) == 1, f"expected one table row for {check_id}, found {len(rows)}"
    return rows[0]


def _src(key, written, embed, stem=SCENARIO):
    """The ``src`` value of the player of file ``key``: its data URI (embed) or relative path (link)."""
    if embed:
        return DATA_URI + base64.b64encode(written[key]).decode("ascii")
    return _audio_path(stem, key)


def _player_srcs(segment):
    return re.findall(r"""<audio\b[^>]*\ssrc\s*=\s*["']([^"']*)["']""", segment)


def _timelines(segment):
    return TIMELINE_SVG.findall(segment)


def _one_timeline(segment):
    found = _timelines(segment)
    assert len(found) == 1, f'expected one <svg class="timeline"> in the scenario, found {len(found)}'
    return found[0]


def _titled_rects(svg):
    """``(classes, title)`` of every rect that has a ``<title>`` (legend swatches without one are ignored)."""
    found = []
    for attrs, title in TITLED_RECT.findall(svg):
        match = CLASS_ATTR.search(attrs)
        classes = set(match.group(1).split()) if match else set()
        found.append((classes, html_module.unescape(title).strip()))
    return found


# =============================================================================================
# B1: full call recording
# =============================================================================================


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_recording_block_and_player_come_before_the_transcript(run_builder, tmp_path, embed):
    run, written = _voice_run(run_builder, tmp_path)
    page = _render(run, tmp_path, embed)
    segment = _scenario_segment(page)
    assert segment.count(RECORDING_DIV) == 1
    recording_src = _src("recording", written, embed)
    srcs = _player_srcs(segment)
    assert recording_src in srcs, "the recording has a player"
    div_at = segment.index(RECORDING_DIV)
    player_at = segment.index(recording_src)
    assert div_at < player_at
    assert player_at < segment.index('<ol class="turns">')
    for heading in ("<h3>Assertions</h3>", "<h3>Metrics</h3>", "<h3>Transcript</h3>"):
        assert player_at < segment.index(heading)
    # The player directly follows the label.
    assert re.match(r"\s*<audio\b", segment[div_at + len(RECORDING_DIV):])
    if not embed:
        assert DATA_URI not in page


def test_recording_block_comes_before_the_inject_note_and_the_error_box(run_builder, tmp_path):
    b = run_builder
    _write_audio(tmp_path)
    result = b.scenario_result(
        SCENARIO,
        transcript=_voice_transcript(b),
        error="LiveKit agent did not finish its turn within 30 s.",
        inject_events=[{"at_turn": 1, "type": "silence", "silence_ms": 2000}],
    )
    page = _render(b.run([result], voice_mode=True), tmp_path, True)
    segment = _scenario_segment(page)
    div_at = segment.index(RECORDING_DIV)
    assert div_at < segment.index(INJECT_NOTE)
    assert div_at < segment.index('class="error-box"')


def test_no_recording_block_without_recording_path(run_builder, tmp_path):
    b = run_builder
    _write_audio(tmp_path)
    result = b.scenario_result(SCENARIO, transcript=_voice_transcript(b, recording=False))
    page = _render(b.run([result], voice_mode=True), tmp_path, True)
    assert "Full call recording" not in page
    assert 'class="recording"' not in page


@pytest.mark.parametrize("path", ["../secret.wav", "audio/notes.txt"])
def test_recording_path_follows_the_turn_audio_refusal_rules(run_builder, tmp_path, path):
    b = run_builder
    _write_audio(tmp_path)
    (tmp_path / "secret.wav").write_bytes(_tone_wav(900))
    transcript = _voice_transcript(b)
    transcript.recording_path = path
    run = b.run([b.scenario_result(SCENARIO, transcript=transcript)], voice_mode=True)
    for embed in (True, False):
        segment = _scenario_segment(_render(run, tmp_path, embed))
        assert RECORDING_DIV in segment
        tail = segment[segment.index(RECORDING_DIV) + len(RECORDING_DIV):]
        assert not re.match(r"\s*<audio\b", tail), f"{path!r} must not get a player (embed={embed})"
        assert html_module.escape(path, quote=True) in tail.split("</div>", 1)[0] + "</div>"
    assert DATA_URI + base64.b64encode(_tone_wav(900)).decode("ascii") not in _render(run, tmp_path, True)


# =============================================================================================
# B2: a player on every turn with audio_path, user turns too
# =============================================================================================


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_user_and_agent_turns_each_have_their_player(run_builder, tmp_path, embed):
    run, written = _voice_run(run_builder, tmp_path)
    page = _render(run, tmp_path, embed)
    for number, (speaker, _text, _start, _end, _latency, key) in enumerate(TURNS, start=1):
        item = _turn_segment(page, number)
        srcs = _player_srcs(item)
        if key is None:
            assert srcs == [], f"turn {number} has no audio file"
            continue
        assert srcs == [_src(key, written, embed)], f"turn {number} ({speaker})"
        assert item.index('class="turn-text"') < item.index(srcs[0]), "the player follows the turn's text"


# =============================================================================================
# B3: players next to failed checks
# =============================================================================================


def _hear_link(number, scenario_index=1):
    """The "Hear turn N" in-page link to turn N's player."""
    return re.compile(
        rf"""<a\b[^>]*\bhref\s*=\s*["']#scenario-{scenario_index}-turn-{number}-audio["'][^>]*>\s*Hear turn {number}\s*</a>"""
    )


def _assert_player_target(page, number, src):
    """Exactly one element of the page carries ``id="scenario-1-turn-N-audio"``; it is inside turn N's
    transcript item and holds that turn's player (the target comes before the player's ``src``)."""
    target = f'id="scenario-1-turn-{number}-audio"'
    assert page.count(target) == 1, f"one player target for turn {number}"
    item = _turn_segment(page, number)
    assert target in item, f"the target is in turn {number}'s transcript item"
    assert item.index(target) < item.index(src), "the target holds the turn's player"


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_failed_assertion_row_links_to_the_turn_player(run_builder, tmp_path, embed):
    run, written = _voice_run(run_builder, tmp_path)
    page = _render(run, tmp_path, embed)
    row = _row_for(_scenario_segment(page), "dpdp.consent_before_pii")
    assert "<audio" not in row, "no second player next to the failed check"
    link = _hear_link(3).search(row)
    assert link, f"the row links to turn 3's player: {row}"
    assert row.index('href="#scenario-1-turn-3"') < link.start(), "the link follows the turn link"
    _assert_player_target(page, 3, _src("t3_user", written, embed))


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_poor_metric_row_links_to_the_turn_player(run_builder, tmp_path, embed):
    run, written = _voice_run(run_builder, tmp_path)
    page = _render(run, tmp_path, embed)
    row = _row_for(_scenario_segment(page), "UNIV-001")
    assert "<audio" not in row, "no second player next to the poor metric"
    link = _hear_link(4).search(row)
    assert link, f"the row links to turn 4's player: {row}"
    assert row.index('href="#scenario-1-turn-4"') < link.start(), "the link follows the turn link"
    _assert_player_target(page, 4, _src("t4_agent", written, embed))


def test_every_turn_player_is_a_link_target(run_builder, tmp_path):
    run, written = _voice_run(run_builder, tmp_path)
    page = _render(run, tmp_path, True)
    for number, (_speaker, _text, _start, _end, _latency, key) in enumerate(TURNS, start=1):
        if key is None:
            assert f'id="scenario-1-turn-{number}-audio"' not in page, f"turn {number} has no audio file"
        else:
            _assert_player_target(page, number, _src(key, written, True))


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
@pytest.mark.parametrize(
    "check_id",
    [
        "dpdp.purpose_disclosure",         # passed
        "dpdp.withdrawal_path",            # skipped
        "rbi_kyc.no_credential_request",   # failed, but its turn has no audio
        "UNIV-013",                        # warning
        "UNIV-003",                        # good
        "VOICE-011",                       # poor but informational
        "UNIV-023",                        # skipped
    ],
)
def test_other_check_rows_have_no_player(run_builder, tmp_path, embed, check_id):
    run, _ = _voice_run(run_builder, tmp_path)
    segment = _scenario_segment(_render(run, tmp_path, embed))
    row = _row_for(segment, check_id)
    assert "<audio" not in row
    assert "Hear turn" not in row and "-audio" not in row


# =============================================================================================
# B4: static SVG timeline
# =============================================================================================


def _default_titles():
    return {
        f"turn 1, caller, 0{DASH}1800 ms",
        f"turn 2, agent, 2400{DASH}4000 ms",
        f"turn 3, caller, 4000{DASH}6000 ms",
        f"turn 4, agent, 8000{DASH}9500 ms",
        f"turn 5, caller, 9500{DASH}10500 ms",
    }


def test_timeline_once_per_voice_scenario_before_the_transcript(run_builder, tmp_path):
    b = run_builder
    text_turns = [
        b.turn("user", "Hello.", 0, end_ms=500),
        b.turn("agent", "Hi.", 900, end_ms=1500, latency_ms=400.0),
    ]
    text_result = b.scenario_result("text_call", transcript=b.transcript("text_call", text_turns))
    no_end_turns = [
        b.turn("user", "Hello.", 0, audio_path="audio/booking_t1_user.wav"),
        b.turn("agent", "Hi.", 900, latency_ms=400.0),
    ]
    no_end = b.scenario_result("no_end_ms", transcript=b.transcript("no_end_ms", no_end_turns, voice_mode=True))
    run, _ = _voice_run(b, tmp_path, extra=[text_result, no_end])
    page = _render(run, tmp_path, True)

    voice = _scenario_segment(page, 1)
    assert len(_timelines(voice)) == 1
    assert voice.index(_timelines(voice)[0]) < voice.index('<ol class="turns">')
    assert _timelines(_scenario_segment(page, 2)) == [], "no timeline for a text-mode scenario"
    assert _timelines(_scenario_segment(page, 3)) == [], "no timeline without any end_ms"
    assert len(TIMELINE_SVG.findall(page)) == 1


def test_timeline_rects_with_default_thresholds(run_builder, tmp_path):
    run, _ = _voice_run(run_builder, tmp_path)   # no metadata["impulse_voice"]: defaults 1500 / 1500 / 3000
    svg = _one_timeline(_scenario_segment(_render(run, tmp_path, False)))
    assert "caller" in svg and "agent" in svg
    rects = _titled_rects(svg)
    titles = {title for _, title in rects}
    assert _default_titles() <= titles, sorted(titles)

    dead_air = [title for classes, title in rects if "dead-air" in classes]
    assert len(dead_air) == 2, dead_air
    assert any(title.startswith("dead air before turn 4: 2000") for title in dead_air), dead_air
    assert any(title.startswith("dead air before turn 6: 3500") for title in dead_air), dead_air

    slow = [title for classes, title in rects if "slow" in classes]
    assert len(slow) == 1, slow
    assert "turn 6" in slow[0]
    assert any(title.startswith("slow reply, turn 6: 3500") for title in titles), sorted(titles)
    assert not any("turn 4" in title for classes, title in rects if "slow" in classes)


def test_timeline_uses_the_stored_thresholds(run_builder, tmp_path):
    thresholds = {"dead_air_ms": 2000, "ttfa_warn_ms": 1000, "ttfa_fail_ms": 1800}
    run, _ = _voice_run(run_builder, tmp_path, metadata={"impulse_voice": {"thresholds": thresholds, "turns": []}})
    svg = _one_timeline(_scenario_segment(_render(run, tmp_path, False)))
    rects = _titled_rects(svg)
    dead_air = [title for classes, title in rects if "dead-air" in classes]
    # turn 4: latency 2000 is not > 2000; turn 6: 3500 > 2000.
    assert len(dead_air) == 1 and dead_air[0].startswith("dead air before turn 6"), dead_air
    slow = [title for classes, title in rects if "slow" in classes]
    # turn 4: 2000 > 1800 and turn 6: 3500 > 1800; turn 2: 600 is not.
    assert len(slow) == 2, slow
    assert any("turn 4" in title for title in slow) and any("turn 6" in title for title in slow)


def test_timeline_leaves_out_turns_without_end_ms(run_builder, tmp_path):
    b = run_builder
    _write_audio(tmp_path)
    transcript = _voice_transcript(b)
    transcript.turns[2].end_ms = None
    run = b.run([b.scenario_result(SCENARIO, transcript=transcript)], voice_mode=True)
    svg = _one_timeline(_scenario_segment(_render(run, tmp_path, False)))
    titles = {title for _, title in _titled_rects(svg)}
    assert not any(title.startswith("turn 3,") for title in titles), sorted(titles)
    assert f"turn 1, caller, 0{DASH}1800 ms" in titles


def test_timeline_is_static_and_escaped(run_builder, tmp_path):
    run, _ = _voice_run(run_builder, tmp_path, name=f"booking {SCRIPT}")
    page = _render(run, tmp_path, True)
    svg = _one_timeline(_scenario_segment(page))
    assert "<script" not in svg.lower()
    assert "href" not in svg.lower()
    assert not re.search(r"\bon[a-z]+\s*=", svg, re.IGNORECASE), "no event handler attributes"
    for title in re.findall(r"<title>(.*?)</title>", svg, re.DOTALL):
        assert "<" not in title, f"unescaped markup in an SVG title: {title!r}"


# =============================================================================================
# Runner: thresholds stored from metrics.voice (voice mode)
# =============================================================================================

VOICE_CONFIG = {
    "tts": {"provider": "openai_compatible", "model": "impulse-test-tts", "base_url": "http://localhost:8000/v1"},
    "stt": {"provider": "openai_compatible", "model": "impulse-test-stt", "base_url": "http://localhost:8000/v1"},
    "save_audio": False,
    "save_recording": False,
    "timeout_s": 30,
}


async def _voice_runner_result(project, llm_router, fake_adapter, metrics):
    from conftest import pcm_tone, wav_bytes

    from impulse_evals.runner import run

    fake_adapter.replies["thresholds_call"] = [
        {"text": "Sure.", "audio_bytes": wav_bytes(pcm_tone(0.4, 16000, frequency_hz=330.0)), "latency_ms": 300.0},
    ]
    project.write_config(
        [project.scenario("thresholds_call", max_turns=1, opening_user_message="please book a slot on tuesday",
                          metrics=["VOICE-029"])],
        voice_config=dict(VOICE_CONFIG),
        metrics=metrics,
    )
    result = await run(project.load(), adapter_factory=fake_adapter)
    scenario_result = result.scenario_results[0]
    assert scenario_result.error is None
    return scenario_result.transcript.metadata


@pytest.mark.asyncio
async def test_runner_stores_the_configured_voice_thresholds(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    metrics = {"enabled": True, "voice": {"dead_air_ms": 1200, "ttfa_warn_ms": 900, "ttfa_fail_ms": 2500}}
    metadata = await _voice_runner_result(impulse_project, llm_router, fake_adapter, metrics)
    assert metadata["impulse_voice"]["thresholds"] == {"dead_air_ms": 1200, "ttfa_warn_ms": 900, "ttfa_fail_ms": 2500}
    assert "turns" in metadata["impulse_voice"]   # the existing per-turn data stays


@pytest.mark.asyncio
async def test_runner_stores_the_default_voice_thresholds(
    impulse_project, llm_router, fake_adapter, mock_voice_server
):
    metadata = await _voice_runner_result(impulse_project, llm_router, fake_adapter, {"enabled": True})
    assert metadata["impulse_voice"]["thresholds"] == {"dead_air_ms": 1500, "ttfa_warn_ms": 1500, "ttfa_fail_ms": 3000}


# =============================================================================================
# Security and size
# =============================================================================================


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_report_stays_script_free_escaped_and_without_external_urls(run_builder, tmp_path, embed):
    run, _ = _voice_run(run_builder, tmp_path, name=f"booking {SCRIPT}")
    page = _render(run, tmp_path, embed)
    assert "<script" not in page.lower()
    assert "alert(9)" in page   # the transcript text is shown, escaped
    assert not EXTERNAL_ATTR.search(page), EXTERNAL_ATTR.search(page).group(0)
    assert not EXTERNAL_CSS.search(page), EXTERNAL_CSS.search(page).group(0)
    assert '<link rel="stylesheet"' not in page.lower()


def test_embedded_size_warning_counts_user_agent_and_recording_files(
    run_builder, tmp_path, capsys, normalize_console
):
    b = run_builder
    folder = tmp_path / "audio"
    folder.mkdir()
    size = 17 * 1024 * 1024   # 3 x 17 MiB = 51 MiB > 50; the agent file alone is not
    for name in ("big_t1_user.wav", "big_t2_agent.wav", "big_recording.wav"):
        _write_large_wav(folder / name, size)
    turns = [
        b.turn("user", "Hello.", 0, end_ms=1000, audio_path="audio/big_t1_user.wav"),
        b.turn("agent", "Hi.", 1500, end_ms=2500, latency_ms=500.0, audio_path="audio/big_t2_agent.wav"),
    ]
    transcript = b.transcript("big", turns, voice_mode=True, recording_path="audio/big_recording.wav")
    run = b.run([b.scenario_result("big", transcript=transcript)], voice_mode=True)
    _render(run, tmp_path, True)
    captured = capsys.readouterr()
    match = SIZE_WARNING.search(normalize_console(captured.out + "\n" + captured.err))
    assert match, "the warning counts the user, agent and recording files"
    assert match.group(1) == "68.0"   # fixes round 4 (O4): base64 of 3 x 17 MiB (superseded: 51.0 raw bytes)


def test_size_warning_counts_a_shared_file_once_per_player(run_builder, tmp_path, capsys, normalize_console):
    # B-rev (supersedes "counts once"): two turns naming the same 30 MiB file embed it twice: 60 MiB.
    b = run_builder
    folder = tmp_path / "audio"
    folder.mkdir()
    _write_large_wav(folder / "shared.wav", 30 * 1024 * 1024)
    turns = [
        b.turn("user", "Hello.", 0, end_ms=1000, audio_path="audio/shared.wav"),
        b.turn("agent", "Hi.", 1500, end_ms=2500, latency_ms=500.0, audio_path="audio/shared.wav"),
    ]
    run = b.run([b.scenario_result("shared", transcript=b.transcript("shared", turns, voice_mode=True))],
                voice_mode=True)
    _render(run, tmp_path, True)
    captured = capsys.readouterr()
    match = SIZE_WARNING.search(normalize_console(captured.out + "\n" + captured.err))
    assert match, "both player copies of the shared file are counted"
    assert match.group(1) == "80.0"   # fixes round 4 (O4): base64 of 2 x 30 MiB (superseded: 60.0 raw bytes)


def test_failed_checks_add_no_embedded_copy(run_builder, tmp_path, capsys, normalize_console):
    """One 51 MiB agent turn named by a failed assertion and a poor metric is embedded once (its
    transcript player); the two rows only link to it. 51 MiB counted once (> 50): 51.0, not 153.0."""
    b = run_builder
    folder = tmp_path / "audio"
    folder.mkdir()
    _write_large_wav(folder / "copies_t2_agent.wav", 51 * 1024 * 1024)
    turns = [
        b.turn("user", "Hello.", 0, end_ms=1000),
        b.turn("agent", "Hi.", 1500, end_ms=2500, latency_ms=500.0, audio_path="audio/copies_t2_agent.wav"),
    ]
    result = b.scenario_result(
        "copies",
        transcript=b.transcript("copies", turns, voice_mode=True),
        assertions=[("dpdp.consent_before_pii", "fail", {"turn_index": 2})],
        metrics=[("UNIV-001", 0.0, {"turn_index": 2})],
    )
    page = _render(b.run([result], voice_mode=True), tmp_path, True)
    uri = DATA_URI + base64.b64encode((folder / "copies_t2_agent.wav").read_bytes()).decode("ascii")
    assert page.count(uri) == 1, "only the transcript player embeds the file"
    assert len(_hear_link(2).findall(page)) == 2, "the assertion row and the metric row link to it"
    captured = capsys.readouterr()
    match = SIZE_WARNING.search(normalize_console(captured.out + "\n" + captured.err))
    assert match, "the one real copy (51 MiB) is over 50 MB"
    assert match.group(1) == "68.0"   # fixes round 4 (O4): base64 of one 51 MiB copy (superseded: 51.0 raw bytes)


def test_size_warning_counts_only_real_embedded_copies(run_builder, tmp_path, capsys, normalize_console):
    """Two 17 MiB turns, each named by failed checks: 34 MiB embedded (the links add nothing), so no
    warning; the old per-player count (2 transcript players + 3 check players = 85 MiB) warned."""
    b = run_builder
    folder = tmp_path / "audio"
    folder.mkdir()
    size = 17 * 1024 * 1024
    for name in ("real_t1_user.wav", "real_t2_agent.wav"):
        _write_large_wav(folder / name, size)
    turns = [
        b.turn("user", "Hello.", 0, end_ms=1000, audio_path="audio/real_t1_user.wav"),
        b.turn("agent", "Hi.", 1500, end_ms=2500, latency_ms=500.0, audio_path="audio/real_t2_agent.wav"),
    ]
    result = b.scenario_result(
        "real",
        transcript=b.transcript("real", turns, voice_mode=True),
        assertions=[
            ("dpdp.consent_before_pii", "fail", {"turn_index": 1}),
            ("rbi_kyc.no_credential_request", "fail", {"turn_index": 2, "severity": "high"}),
        ],
        metrics=[("UNIV-001", 0.0, {"turn_index": 2})],
    )
    page = _render(b.run([result], voice_mode=True), tmp_path, True)
    assert len(_player_srcs(page)) == 2, "one player per turn, none in the check rows"
    captured = capsys.readouterr()
    assert not SIZE_WARNING.search(normalize_console(captured.out + "\n" + captured.err)), (
        "34 MiB of real copies is under 50 MB"
    )


# =============================================================================================
# B-rev: odd timeline values never stop the report; that turn is left out of the timeline
# =============================================================================================


def _render_from_run_json(run, run_dir, mutate):
    """``run`` written as run.json text, changed by ``mutate(turns)`` (dicts), loaded back and rendered."""
    import json

    from impulse_evals.models import RunResult

    data = json.loads(run.model_dump_json())
    mutate(data["scenario_results"][0]["transcript"]["turns"])
    loaded = RunResult.model_validate_json(json.dumps(data))
    return _render(loaded, run_dir, False)


def _assert_turn_3_left_out(page):
    svg = _one_timeline(_scenario_segment(page))
    titles = {title for _, title in _titled_rects(svg)}
    assert not any(title.startswith("turn 3,") for title in titles), sorted(titles)
    assert f"turn 1, caller, 0{DASH}1800 ms" in titles
    assert f"turn 4, agent, 8000{DASH}9500 ms" in titles
    bad = re.search(r"\b-?(?:nan|inf(?:inity)?)\b", svg, re.IGNORECASE)
    assert bad is None, f"{bad.group(0)!r} written into the timeline"


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("end_ms", 10**400, id="end-10e400"),
        pytest.param("timestamp_ms", 10**400, id="start-10e400"),
        pytest.param("end_ms", 10**13, id="end-over-10e12"),
        pytest.param("end_ms", -(10**13), id="end-below-minus-10e12"),
    ],
)
def test_timeline_leaves_out_a_turn_with_an_out_of_range_time_in_run_json(run_builder, tmp_path, field, value):
    b = run_builder
    _write_audio(tmp_path)
    run = b.run([b.scenario_result(SCENARIO, transcript=_voice_transcript(b))], voice_mode=True)

    def mutate(turns):
        turns[2][field] = value

    _assert_turn_3_left_out(_render_from_run_json(run, tmp_path, mutate))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", ["end_ms", "timestamp_ms"])
def test_timeline_leaves_out_a_turn_with_a_non_finite_time(run_builder, tmp_path, field, value):
    # Turn times are ints in run.json (pydantic refuses NaN there); a NaN-like value can still reach
    # render_html on a result built in code (assignment is not validated).
    b = run_builder
    _write_audio(tmp_path)
    transcript = _voice_transcript(b)
    setattr(transcript.turns[2], field, value)
    run = b.run([b.scenario_result(SCENARIO, transcript=transcript)], voice_mode=True)
    _assert_turn_3_left_out(_render(run, tmp_path, False))

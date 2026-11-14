# SPDX-License-Identifier: AGPL-3.0-only
"""Fixes round 4, L8, L22, L25 (FINAL-TEST.md).

* L8: ``config.URL_CREDENTIAL_QUERY_KEYS`` also holds ``apikey``, ``api_token``, ``sig``, ``signature``,
  ``secret``, ``password`` and ``auth``: ``strip_url_credentials`` removes such query parameters and
  ``url_credentials`` returns their values (as written and decoded) for redaction.
* L22: ``pyproject.toml`` pins ``jiwer>=4`` (jiwer 3.x raises on an empty reference; 4.x counts insertions).
* L25: the HTML report never carries an absolute path: an absolute ``audio_path`` is refused in embed mode
  too (as link mode already did) and the note shows only the file name; a file that cannot be read gives
  a note with the OS error text (``strerror``) or the exception type, never ``str(exc)`` (which holds the
  absolute file name).
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW_NAMES = ("apikey", "api_token", "sig", "signature", "secret", "password", "auth")
OLD_NAMES = ("token", "key", "api_key", "api-key", "access_token")
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]|(?<![\w/.:])/(?:Users|home|tmp|var)/")


# -- L8 ----------------------------------------------------------------------------------------


def test_credential_query_key_names():
    from impulse_evals.config import URL_CREDENTIAL_QUERY_KEYS

    for name in OLD_NAMES + NEW_NAMES:
        assert name in URL_CREDENTIAL_QUERY_KEYS, name


@pytest.mark.parametrize("name", NEW_NAMES)
def test_new_credential_query_parameters_are_stripped_and_known(name):
    from impulse_evals.config import strip_url_credentials, url_credentials

    url = f"wss://bot.example/ws?{name}=sk-FAKE-q-0123&room=r1"
    assert strip_url_credentials(url) == "wss://bot.example/ws?room=r1"
    assert "sk-FAKE-q-0123" in url_credentials(url)
    upper = f"https://api.example/v1?{name.upper()}=sk-FAKE-upper-0123"
    assert "sk-FAKE-upper-0123" in url_credentials(upper)
    assert "sk-FAKE-upper-0123" not in strip_url_credentials(upper)


def test_non_credential_query_parameters_are_kept():
    from impulse_evals.config import strip_url_credentials, url_credentials

    url = "https://api.example/v1?room=r1&author=me&signal=on&secretary=x"
    assert strip_url_credentials(url) == url
    assert url_credentials(url) == []


# -- L22 ---------------------------------------------------------------------------------------


def test_pyproject_pins_jiwer_4():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    jiwer = [d for d in deps if d.split(">")[0].split("=")[0].strip().lower() == "jiwer"]
    assert jiwer == ["jiwer>=4"], deps


# -- L25 ---------------------------------------------------------------------------------------


def _wav(path):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\0\0" * 400)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def _voice_run(run_builder, audio_path):
    b = run_builder
    turns = [
        b.turn("user", "Hello there.", 0),
        b.turn("agent", "Hi, how can I help?", 500, latency_ms=500.0, audio_path=audio_path, end_ms=1500),
    ]
    transcript = b.transcript("booking", turns, voice_mode=True)
    return b.run([b.scenario_result("booking", transcript=transcript)], voice_mode=True)


@pytest.mark.parametrize("embed", [True, False], ids=["embed", "link"])
def test_absolute_audio_path_is_refused_and_shown_as_a_file_name_only(run_builder, tmp_path, embed):
    from impulse_evals.reporter import render_html

    file = tmp_path / "audio" / "booking_t2_agent.wav"
    _wav(file)
    run = _voice_run(run_builder, str(file))   # absolute, inside <run_dir>/audio/
    html = render_html(run, embed_audio=embed, run_dir=tmp_path)
    assert "data:audio/wav;base64," not in html
    assert "booking_t2_agent.wav" in html
    assert str(tmp_path) not in html and str(tmp_path).replace("\\", "/") not in html
    assert not ABSOLUTE_PATH.search(html), ABSOLUTE_PATH.search(html).group(0)


@pytest.mark.parametrize(
    "audio_path, shown",
    [
        pytest.param("file:///C:/Users/someone/run/audio/x.wav", "x.wav", id="file-scheme"),
        pytest.param("https://host.example/runs/audio/x.wav", "x.wav", id="https-scheme"),
        pytest.param("\\\\server\\share\\audio\\x.wav", "x.wav", id="unc"),
        pytest.param("D:\\runs\\audio\\x.wav", "x.wav", id="drive-backslash"),
        pytest.param("audio/../x.wav", "audio/../x.wav", id="relative-dotdot-unchanged"),
    ],
)
def test_scheme_unc_and_drive_paths_are_shown_by_their_file_name(audio_path, shown):
    """Stage D fix-up (round 4 review, finding 3): a URL-style path is refused like an absolute one and
    its note shows only the last segment."""
    from impulse_evals.reporter import _shown_path

    assert _shown_path(audio_path) == shown


def test_unreadable_audio_file_note_has_no_absolute_path(run_builder, tmp_path, monkeypatch):
    from impulse_evals.reporter import render_html

    file = tmp_path / "audio" / "booking_t2_agent.wav"
    _wav(file)
    real_read_bytes = Path.read_bytes

    def failing_read_bytes(self):
        if self.name == "booking_t2_agent.wav":
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    run = _voice_run(run_builder, "audio/booking_t2_agent.wav")
    html = render_html(run, embed_audio=True, run_dir=tmp_path)
    assert "audio file could not be read: audio/booking_t2_agent.wav (Permission denied)" in html
    assert str(tmp_path) not in html and str(tmp_path).replace("\\", "/") not in html
    assert not ABSOLUTE_PATH.search(html)


def test_unreadable_audio_file_without_strerror_shows_the_exception_type(run_builder, tmp_path, monkeypatch):
    from impulse_evals.reporter import render_html

    file = tmp_path / "audio" / "booking_t2_agent.wav"
    _wav(file)

    def failing_read_bytes(self):
        raise OSError(str(self))   # no strerror (errno-less): the type name is shown

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    run = _voice_run(run_builder, "audio/booking_t2_agent.wav")
    html = render_html(run, embed_audio=True, run_dir=tmp_path)
    assert "audio file could not be read: audio/booking_t2_agent.wav (OSError)" in html
    assert str(tmp_path) not in html

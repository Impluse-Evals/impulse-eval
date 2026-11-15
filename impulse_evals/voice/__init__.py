# SPDX-License-Identifier: Apache-2.0
"""Voice pipeline providers (SPEC.md section 14).

``voice.tts`` / ``voice.stt`` hold the ``TTSProvider`` / ``STTProvider`` interfaces and the factories
``get_tts_provider(cfg)`` / ``get_stt_provider(cfg)``; ``local_hf``, ``local_light`` (Kokoro ONNX +
faster-whisper, no PyTorch), ``openai_compatible``, ``impulse_hosted`` and ``litellm``
(``litellm_voice``) are the providers (``config.VOICE_PROVIDERS`` lists every provider name);
``audio_utils`` reads and writes WAV audio.

Heavy optional libraries (transformers, faster-whisper, kokoro-onnx, onnxruntime, torch, pydub, ...)
are imported only inside the functions that use them, so importing this package or any module in it
never needs an extra. This package module imports no other Impulse module at import time.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TIMEOUT_S = 30.0


class VoiceError(RuntimeError):
    """A voice provider could not do its job (request failed, unreadable audio, unsupported language)."""


def primary_subtag(language: str) -> str:
    """The primary language subtag, for engines that expect a bare language (``hi-IN`` -> ``hi``)."""
    return language.strip().split("-", 1)[0].lower()


def provider_block(cfg: Any, side: str) -> tuple[Any, float]:
    """``(sub_block, timeout_s)`` for ``side`` (``"tts"`` or ``"stt"``) of a voice configuration.

    ``cfg`` is the loaded ``voice_config`` block (``VoiceConfig``), a whole ``ProjectConfig`` (its
    ``voice_config`` is used), or the ``tts`` / ``stt`` sub-block itself (timeout then 30 s).
    """
    if hasattr(cfg, "voice_config") and not hasattr(cfg, side):
        if cfg.voice_config is None:
            from impulse_evals.config import ConfigError

            raise ConfigError("voice_config", "voice_config is not set; voice providers need it")
        cfg = cfg.voice_config
    if hasattr(cfg, side) and hasattr(cfg, "timeout_s"):
        timeout = cfg.timeout_s
        return getattr(cfg, side), float(timeout) if timeout else DEFAULT_TIMEOUT_S
    return cfg, DEFAULT_TIMEOUT_S

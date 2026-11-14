# SPDX-License-Identifier: AGPL-3.0-only
"""Text to speech: the ``TTSProvider`` interface and the ``get_tts_provider`` factory (SPEC.md section 14).

``get_tts_provider(cfg)`` takes the loaded ``voice_config`` block (``ProjectConfig.voice_config``)
and returns the provider named by ``voice_config.tts.provider``. Provider modules are imported only
when selected, and no provider imports a heavy library before it is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from impulse_evals.config import VOICE_PROVIDERS, ConfigError
from impulse_evals.voice import provider_block

if TYPE_CHECKING:
    from impulse_evals.config import TomlConfig


class TTSProvider(ABC):
    """Synthesizes speech. ``synthesize`` returns WAV bytes."""

    name: str = ""

    @abstractmethod
    async def synthesize(self, text: str, language: str, voice: Optional[str]) -> bytes:
        """Speak ``text`` in ``language`` (a full code such as ``hi-IN``) with ``voice`` (None = default)."""

    def check_language(self, language: str) -> None:
        """Raise before any request when this provider cannot speak ``language``: ``VoiceError`` for an
        unsupported language, ``ConfigError`` for a missing setting (e.g. no model for it). The
        runner calls it for every scenario language before the run starts. Default: no limit."""


def get_tts_provider(cfg: Any, *, toml: Optional[TomlConfig] = None) -> TTSProvider:
    """The TTS provider for ``cfg`` (the ``voice_config`` block; see ``voice.provider_block``).

    ``toml`` (the loaded ``.impulse/config.toml``) is consulted for keys after the environment.
    An unknown provider is a ``ConfigError`` at ``voice_config.tts.provider``; ``litellm`` with a key
    that is not set is a ``ConfigError`` at ``voice_config.tts.model`` (raised here, before any call).
    """
    block, timeout_s = provider_block(cfg, "tts")
    provider = block.provider
    if provider == "openai_compatible":
        from impulse_evals.voice.openai_compatible import OpenAICompatibleTTS

        return OpenAICompatibleTTS(block, timeout_s=timeout_s, toml=toml)
    if provider == "local_hf":
        from impulse_evals.voice.local_hf import LocalHFTTS

        return LocalHFTTS(block)
    if provider == "local_light":
        from impulse_evals.voice.local_light import LocalLightTTS

        return LocalLightTTS(block)
    if provider == "impulse_hosted":
        from impulse_evals.voice.impulse_hosted import ImpulseHostedTTS

        return ImpulseHostedTTS(block, timeout_s=timeout_s, toml=toml)
    if provider == "litellm":
        from impulse_evals.voice.litellm_voice import LiteLLMTTS

        return LiteLLMTTS(block, timeout_s=timeout_s, toml=toml)
    raise ConfigError(
        "voice_config.tts.provider", f"unknown provider {provider}. Use one of: {', '.join(VOICE_PROVIDERS)}."
    )

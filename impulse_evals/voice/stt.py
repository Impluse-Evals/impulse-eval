# SPDX-License-Identifier: Apache-2.0
"""Speech to text: the ``STTProvider`` interface and the ``get_stt_provider`` factory (SPEC.md section 14).

``get_stt_provider(cfg)`` takes the loaded ``voice_config`` block (``ProjectConfig.voice_config``)
and returns the provider named by ``voice_config.stt.provider``. Provider modules are imported only
when selected, and no provider imports a heavy library before it is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from impulse_evals.config import VOICE_PROVIDERS, ConfigError
from impulse_evals.models import STTResult
from impulse_evals.voice import provider_block

if TYPE_CHECKING:
    from impulse_evals.config import TomlConfig


class STTProvider(ABC):
    """Transcribes speech. ``transcribe`` takes WAV bytes."""

    name: str = ""

    @abstractmethod
    async def transcribe(self, audio: bytes, language: str) -> STTResult:
        """Transcribe ``audio`` spoken in ``language`` (a full code such as ``hi-IN``)."""

    def check_language(self, language: str) -> None:
        """Raise before any request when this provider cannot transcribe ``language``: ``VoiceError``
        for an unsupported language, ``ConfigError`` for a missing setting. The runner calls it for
        every scenario language before the run starts. Default: no limit."""


def get_stt_provider(cfg: Any, *, toml: Optional[TomlConfig] = None) -> STTProvider:
    """The STT provider for ``cfg`` (the ``voice_config`` block; see ``voice.provider_block``).

    ``toml`` (the loaded ``.impulse/config.toml``) is consulted for keys after the environment.
    An unknown provider is a ``ConfigError`` at ``voice_config.stt.provider``; ``litellm`` with a key
    that is not set is a ``ConfigError`` at ``voice_config.stt.model`` (raised here, before any call).
    """
    block, timeout_s = provider_block(cfg, "stt")
    provider = block.provider
    if provider == "openai_compatible":
        from impulse_evals.voice.openai_compatible import OpenAICompatibleSTT

        return OpenAICompatibleSTT(block, timeout_s=timeout_s, toml=toml)
    if provider == "local_hf":
        from impulse_evals.voice.local_hf import LocalHFSTT

        return LocalHFSTT(block)
    if provider == "local_light":
        from impulse_evals.voice.local_light import LocalLightSTT

        return LocalLightSTT(block)
    if provider == "impulse_hosted":
        from impulse_evals.voice.impulse_hosted import ImpulseHostedSTT

        return ImpulseHostedSTT(block, timeout_s=timeout_s, toml=toml)
    if provider == "litellm":
        from impulse_evals.voice.litellm_voice import LiteLLMSTT

        return LiteLLMSTT(block, timeout_s=timeout_s, toml=toml)
    raise ConfigError(
        "voice_config.stt.provider", f"unknown provider {provider}. Use one of: {', '.join(VOICE_PROVIDERS)}."
    )

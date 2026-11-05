"""Offline input estimation of complete serialized request views."""

import json
from importlib.metadata import version
from typing import Protocol, runtime_checkable


@runtime_checkable
class InputTokenCounter(Protocol):
    """Count a whole candidate, not a sum of independent message estimates."""

    def count_request(self, request: dict) -> int:
        ...

    def describe(self) -> dict:
        ...


class LocalJsonTokenCounter:
    """Use LiteLLM's installed cl100k BPE as an explicit Qwen surrogate.

    Counts JSON syntax, roles, IDs, arguments, results and tools together. This
    representation is not the provider's native tokenizer or chat framing.
    It never calls model/HuggingFace/remote counting APIs or substitutes zero.
    """

    def count_request(self, request: dict) -> int:
        from litellm.litellm_core_utils.default_encoding import encoding

        for message in request["messages"]:
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError("Local JSON estimator only supports text/tool messages")
        text = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return len(encoding.encode_ordinary(text))

    def describe(self) -> dict:
        return {
            "name": "complete-json-bpe-surrogate",
            "version": "1",
            "tiktoken_version": version("tiktoken"),
            "litellm_version": version("litellm"),
            "tokenizer": "cl100k_base (installed LiteLLM default_encoding)",
            "fallback": "generic local BPE, not Qwen-native tokenization",
            "is_estimate": True,
            "provider_framing_verified": False,
        }

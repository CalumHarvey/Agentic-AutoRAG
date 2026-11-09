"""LiteLLM runtime configuration and model-alias resolution."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import litellm

from agentic_autorag.cost_ledger import get_active_ledger

logger = logging.getLogger(__name__)

DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS = 300.0

# Process-wide alias map. Empty = pure passthrough (every model name is sent
# to LiteLLM verbatim). Populated by ``install_model_aliases``.
_MODEL_ALIASES: dict[str, str | dict[str, Any]] = {}

# Shadow prices for models run locally through Ollama (see
# scripts/build_local_model_prices.py). LiteLLM model id -> (USD per input token,
# USD per output token). Populated by ``install_local_model_prices``.
LOCAL_MODEL_PRICES_PATH = Path(__file__).parent / "local_model_prices.json"
LOCAL_PROVIDERS = ("ollama", "ollama_chat")
# A local model may first have to load from disk before it answers (about 16 s for a
# 7B model on a 6 GB GPU, more when another model must be swapped out), so model
# checks give local models this long instead of an API round trip's timeout.
LOCAL_LOAD_TIMEOUT_S = 180.0
# Marks a LiteLLM catalog entry as a price we assigned, not a listing: such a model
# still gets the live check in ``validate_llm_models``.
SHADOW_PRICE_KEY = "shadow_price_of"
_LOCAL_PRICES: dict[str, tuple[float, float]] = {}


def configure_litellm_runtime(model_aliases: dict[str, Any] | None = None) -> None:
    """Configure LiteLLM for long-running workloads and install model aliases.

    Enables ``drop_params`` so provider-specific parameters are silently
    dropped for models that don't accept them — cross-provider portability
    over strict per-param enforcement.
    """
    os.environ.setdefault(
        "LOGGING_WORKER_MAX_TIME_PER_COROUTINE",
        str(DEFAULT_LOGGING_WORKER_TIMEOUT_SECONDS),
    )
    litellm.drop_params = True
    # Silence LiteLLM's stderr "Provider List" / "Give Feedback" prints. They
    # fire whenever LiteLLM's catalog can't fully identify a model name (e.g.
    # custom Azure deployment names) and add noise without surfacing actionable
    # information — the real errors still raise normally.
    litellm.suppress_debug_info = True
    install_local_model_prices()
    if model_aliases is not None:
        install_model_aliases(model_aliases)


def install_local_model_prices(path: Path = LOCAL_MODEL_PRICES_PATH) -> None:
    """Charge local Ollama models the price of the same model from a hosted provider.

    Reads the table built by ``scripts/build_local_model_prices.py`` and, for each
    Ollama tag, registers ``ollama/<tag>`` and ``ollama_chat/<tag>`` with LiteLLM
    (so code reading ``litellm.model_cost`` sees the price) and with
    ``acompletion_with_cost`` (which charges from it directly). A local model not in
    the table stays at $0. A missing or unreadable table logs a warning and changes
    nothing: local models keep the prices already installed, $0 if none were.
    """
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Local model price table %s unreadable; local models cost $0", path, exc_info=True)
        return
    _LOCAL_PRICES.clear()
    entries: dict[str, dict[str, Any]] = {}
    for tag, price in table.get("models", {}).items():
        per_in = float(price["input_per_1m_usd"]) / 1e6
        per_out = float(price["output_per_1m_usd"]) / 1e6
        for provider in LOCAL_PROVIDERS:
            model_id = f"{provider}/{tag}"
            _LOCAL_PRICES[model_id] = (per_in, per_out)
            entries[model_id] = {
                "input_cost_per_token": per_in,
                "output_cost_per_token": per_out,
                "litellm_provider": provider,
                "mode": "chat",
                SHADOW_PRICE_KEY: price.get("priced_as"),
            }
    litellm.register_model(entries)
    # register_model files ``ollama/<tag>`` under the bare tag; keep the prefixed id
    # too, so a lookup by the id the configs use finds it.
    for model_id, entry in entries.items():
        litellm.model_cost.setdefault(model_id, entry)


def is_local_model(model: str) -> bool:
    """True for a LiteLLM model id served locally through Ollama (``ollama/...``)."""
    return model.split("/", 1)[0] in LOCAL_PROVIDERS and "/" in model


def local_model_price(model: str) -> tuple[float, float] | None:
    """USD per input and output token for a shadow-priced local model, else None."""
    return _LOCAL_PRICES.get(model)


def install_model_aliases(aliases: dict[str, Any]) -> None:
    """Install (or replace) the process-wide model alias map.

    Each entry maps a short name to a LiteLLM target. Two value shapes:
    a plain string ``"provider/deployment"``, or a dict with ``model`` plus
    extra kwargs (``api_base``, ``api_key``, ``api_version``) merged into
    every call — needed for custom OpenAI-compatible endpoints.

    Also registers the resolved target with ``litellm.register_model`` so
    cost lookups inherit the alias key's catalog entry when one exists.
    """
    global _MODEL_ALIASES
    _MODEL_ALIASES = dict(aliases)
    _register_alias_costs()


def _register_alias_costs() -> None:
    for alias, target in _MODEL_ALIASES.items():
        target_model = target if isinstance(target, str) else target.get("model")
        if not target_model or target_model in litellm.model_cost:
            continue
        cost_data = _find_cost_data_for_alias(alias)
        if cost_data is None:
            continue
        entry = dict(cost_data)
        if "/" in target_model:
            entry["litellm_provider"] = target_model.split("/", 1)[0]
        litellm.register_model({target_model: entry})


def _find_cost_data_for_alias(alias: str) -> dict | None:
    if alias in litellm.model_cost:
        return litellm.model_cost[alias]
    if "/" in alias:
        bare = alias.split("/", 1)[1]
        if bare in litellm.model_cost:
            return litellm.model_cost[bare]
    return None


def resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    """Resolve an alias to ``(target_model, extra_kwargs)``.

    Returns ``(model, {})`` unchanged when the name isn't aliased.
    Call-site kwargs always win over the alias's extra kwargs.
    """
    target = _MODEL_ALIASES.get(model)
    if target is None:
        return model, {}
    if isinstance(target, str):
        return target, {}
    extra = dict(target)
    resolved = extra.pop("model")
    return resolved, extra


def _extract_cache_tokens(usage_obj: Any) -> tuple[int, int]:
    """Return ``(cache_read_input_tokens, cache_creation_input_tokens)``.

    Providers report cache reads differently: Anthropic at the top level
    (``cache_read_input_tokens``), OpenAI/Azure-OpenAI under
    ``prompt_tokens_details.cached_tokens``, and DeepSeek's first-party API
    under ``prompt_cache_hit_tokens``. Prefer the top-level Anthropic field,
    then the OpenAI shape, then the DeepSeek shape.
    """
    if usage_obj is None:
        return 0, 0
    cache_creation = int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
    if cache_read == 0:
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            cache_read = int(getattr(details, "cached_tokens", 0) or 0)
    if cache_read == 0:
        cache_read = int(getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0)
    return cache_read, cache_creation


async def acompletion_with_cost(
    *,
    cost_category: str | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, float | int]]:
    """``litellm.acompletion`` wrapper that also returns USD cost and token counts.

    Returns ``(response, {"usd", "prompt_tokens", "completion_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens"})``. ``usd`` falls
    back to 0.0 when LiteLLM has no pricing for the model or the cost call
    raises. When ``cost_category`` is set and a ledger is active, credits the
    call to that bucket.
    """
    original_model = kwargs.pop("model")
    resolved_model, alias_extras = resolve_model(original_model)
    call_kwargs = {**alias_extras, **kwargs, "model": resolved_model}
    response = await litellm.acompletion(**call_kwargs)
    usage_obj = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0
    cache_read, cache_creation = _extract_cache_tokens(usage_obj)
    local_price = local_model_price(resolved_model)
    if local_price is not None:
        # Priced from the table, not completion_cost: Ollama's response does not
        # always carry the provider-prefixed model id the catalog is keyed by.
        usd = prompt_tokens * local_price[0] + completion_tokens * local_price[1]
    else:
        try:
            usd = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            usd = 0.0
            logger.debug("completion_cost failed for model=%s", resolved_model, exc_info=True)
    logger.debug(
        "LLM call: model=%s (alias=%s) category=%s prompt_tokens=%d completion_tokens=%d "
        "cache_read=%d cache_creation=%d cost=$%.6f",
        resolved_model,
        original_model if original_model != resolved_model else "-",
        cost_category or "-",
        prompt_tokens,
        completion_tokens,
        cache_read,
        cache_creation,
        usd,
    )
    if cost_category is not None:
        ledger = get_active_ledger()
        if ledger is not None:
            ledger.record(
                cost_category,
                usd,
                prompt_tokens,
                completion_tokens,
                cache_read,
                cache_creation,
            )
    return response, {
        "usd": usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }

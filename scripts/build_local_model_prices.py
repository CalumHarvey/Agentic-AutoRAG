"""Build agentic_autorag/local_model_prices.json from LiteLLM's price list.

A model run locally through Ollama costs nothing per token, so a cost-aware search
would see every local config as free. Instead each local model is charged a shadow
price: what the same model costs from a hosted provider, taken from LiteLLM's own
price list (``litellm.model_cost``).

For each Ollama tag below, the listings are the LiteLLM ids for that exact model
(same family, size and instruct variant; no fine-tunes, no other distills). The
shadow price is the listing at the median input price, the lower one when the count
is even, ties broken by output price and then id. Input price decides because RAG
prompts are mostly retrieved context (about 750 input tokens per output token in the
smoke trial). Every listing is kept in the file so the choice can be checked.

Run after a LiteLLM upgrade and commit the result:

    uv run python scripts/build_local_model_prices.py
"""

from __future__ import annotations

import json
from datetime import date
from importlib.metadata import version
from pathlib import Path

import litellm

OUT = Path(__file__).resolve().parents[1] / "agentic_autorag" / "local_model_prices.json"

# Ollama tag -> LiteLLM ids of hosted listings of the same model.
LISTINGS: dict[str, list[str]] = {
    "llama3.2:1b": [
        "novita/meta-llama/llama-3.2-1b-instruct",
        "cloudflare/@cf/meta/llama-3.2-1b-instruct",
        "together_ai/meta-llama/Llama-3.2-1B-Instruct",
        "vercel_ai_gateway/meta/llama-3.2-1b",
    ],
    "llama3.2:3b": [
        "lambda_ai/llama3.2-3b-instruct",
        "deepinfra/meta-llama/Llama-3.2-3B-Instruct",
        "novita/meta-llama/llama-3.2-3b-instruct",
        "llamagate/llama-3.2-3b",
        "cloudflare/@cf/meta/llama-3.2-3b-instruct",
        "together_ai/meta-llama/Llama-3.2-3B-Instruct",
        "hyperbolic/meta-llama/Llama-3.2-3B-Instruct",
        "vercel_ai_gateway/meta/llama-3.2-3b",
    ],
    "llama3.1:8b": [
        "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "novita/meta-llama/llama-3.1-8b-instruct",
        "nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "lambda_ai/llama3.1-8b-instruct",
        "nscale/meta-llama/Llama-3.1-8B-Instruct",
        "llamagate/llama-3.1-8b",
        "vercel_ai_gateway/meta/llama-3.1-8b",
        "cerebras/llama3.1-8b",
        "ovhcloud/Llama-3.1-8B-Instruct",
        "hyperbolic/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "perplexity/llama-3.1-8b-instruct",
        "wandb/meta-llama/Llama-3.1-8B-Instruct",
        "snowflake/llama3.1-8b",
    ],
    "qwen2.5:7b": [
        "deepinfra/Qwen/Qwen2.5-7B-Instruct",
        "novita/qwen/qwen2.5-7b-instruct",
        "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    ],
    "qwen3:4b": [
        "novita/qwen/qwen3-4b-fp8",
        "nebius/Qwen/Qwen3-4B",
        "fireworks_ai/accounts/fireworks/models/qwen3-4b",
    ],
    "qwen3:8b": [
        "novita/qwen/qwen3-8b-fp8",
        "llamagate/qwen3-8b",
        "fireworks_ai/accounts/fireworks/models/qwen3-8b",
    ],
    "gemma3:4b": [
        "llamagate/gemma3-4b",
        "google.gemma-3-4b-it",
        "deepinfra/google/gemma-3-4b-it",
    ],
    "phi4-mini": [
        "azure_ai/Phi-4-mini-instruct",
    ],
    "mistral:7b": [  # Ollama's mistral:7b is Mistral-7B-Instruct v0.3
        "ovhcloud/Mistral-7B-Instruct-v0.3",
        "llamagate/mistral-7b-v0.3",
        "fireworks_ai/accounts/fireworks/models/mistral-7b-instruct-v3",
        "together_ai/mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "ministral-3:3b": [
        "mistral/ministral-3-3b-2512",
        "mistral.ministral-3-3b-instruct",
        "fireworks_ai/accounts/fireworks/models/ministral-3-3b-instruct-2512",
    ],
    "ministral-3:8b": [
        "mistral/ministral-3-8b-2512",
        "mistral.ministral-3-8b-instruct",
        "fireworks_ai/accounts/fireworks/models/ministral-3-8b-instruct-2512",
    ],
    "granite3.3:8b": [
        "replicate/ibm-granite/granite-3.3-8b-instruct",
    ],
    "deepseek-r1:7b": [  # DeepSeek-R1-Distill-Qwen-7B
        "gmi/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "fireworks_ai/accounts/fireworks/models/deepseek-r1-distill-qwen-7b",
        "nscale/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ],
    "deepseek-r1:8b": [  # Ollama's deepseek-r1:8b is DeepSeek-R1-0528-Qwen3-8B
        "novita/deepseek/deepseek-r1-0528-qwen3-8b",
        "fireworks_ai/accounts/fireworks/models/deepseek-r1-0528-distill-qwen3-8b",
    ],
}


def _per_million(value: float | None) -> float:
    return round((value or 0.0) * 1e6, 6)


def build() -> dict:
    models: dict[str, dict] = {}
    for tag, ids in LISTINGS.items():
        listings = []
        for model_id in ids:
            info = litellm.model_cost.get(model_id)
            if not info or not info.get("input_cost_per_token"):
                raise SystemExit(f"{tag}: {model_id} is not priced in this LiteLLM; update LISTINGS")
            listings.append(
                {
                    "litellm_id": model_id,
                    "input_per_1m_usd": _per_million(info["input_cost_per_token"]),
                    "output_per_1m_usd": _per_million(info.get("output_cost_per_token")),
                }
            )
        listings.sort(key=lambda x: (x["input_per_1m_usd"], x["output_per_1m_usd"], x["litellm_id"]))
        chosen = listings[(len(listings) - 1) // 2]
        models[tag] = {
            "input_per_1m_usd": chosen["input_per_1m_usd"],
            "output_per_1m_usd": chosen["output_per_1m_usd"],
            "priced_as": chosen["litellm_id"],
            "listings": listings,
        }
    return {
        "description": (
            "Shadow prices for models run locally through Ollama: the hosted listing at the "
            "median input price for the same model in LiteLLM's price list. Built by "
            "scripts/build_local_model_prices.py; applied by litellm_runtime."
        ),
        "litellm_version": version("litellm"),
        "built": date.today().isoformat(),
        "models": models,
    }


def main() -> None:
    table = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(table['models'])} models, LiteLLM {table['litellm_version']})")
    for tag, entry in table["models"].items():
        print(
            f"  {tag:16} ${entry['input_per_1m_usd']:.3f} in / ${entry['output_per_1m_usd']:.3f} out "
            f"per 1M  as {entry['priced_as']}  ({len(entry['listings'])} listings)"
        )


if __name__ == "__main__":
    main()

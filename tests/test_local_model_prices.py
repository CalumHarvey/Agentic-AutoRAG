"""Shadow prices for local Ollama models (litellm_runtime.install_local_model_prices)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from agentic_autorag import litellm_runtime
from agentic_autorag.config import models


def _table(tmp_path: Path) -> Path:
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "testmodel:7b": {
                        "input_per_1m_usd": 0.2,
                        "output_per_1m_usd": 0.5,
                        "priced_as": "someprovider/test-model-7b",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def prices(tmp_path: Path):
    saved = dict(litellm.model_cost)
    litellm_runtime.install_local_model_prices(_table(tmp_path))
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(saved)
    litellm_runtime.install_local_model_prices()


def test_registers_both_ollama_routes(prices) -> None:
    for model_id in ("ollama/testmodel:7b", "ollama_chat/testmodel:7b"):
        assert litellm_runtime.local_model_price(model_id) == pytest.approx((0.2e-6, 0.5e-6))
        entry = litellm.model_cost[model_id]
        assert entry["input_cost_per_token"] == pytest.approx(0.2e-6)
        assert entry["shadow_price_of"] == "someprovider/test-model-7b"


def test_unlisted_local_model_has_no_price(prices) -> None:
    assert litellm_runtime.local_model_price("ollama/not-in-table:1b") is None


def test_shadow_priced_model_still_gets_live_check(prices) -> None:
    assert models._is_in_litellm_catalog("ollama/testmodel:7b") is False
    assert models._is_in_litellm_catalog("ollama_chat/testmodel:7b") is False


def test_shipped_table_loads() -> None:
    litellm_runtime.install_local_model_prices()
    assert litellm_runtime.local_model_price("ollama/qwen2.5:7b") is not None


def test_unreadable_table_keeps_previous_prices(tmp_path: Path) -> None:
    litellm_runtime.install_local_model_prices()
    litellm_runtime.install_local_model_prices(tmp_path / "missing.json")
    try:
        assert litellm_runtime.local_model_price("ollama/qwen2.5:7b") is not None
    finally:
        litellm_runtime.install_local_model_prices()


@pytest.mark.asyncio
async def test_call_is_charged_from_the_table(prices) -> None:
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=100))
    with patch.object(litellm, "acompletion", new=AsyncMock(return_value=response)):
        _, cost = await litellm_runtime.acompletion_with_cost(
            model="ollama/testmodel:7b", messages=[{"role": "user", "content": "hi"}]
        )
    assert cost["usd"] == pytest.approx(1_000 * 0.2e-6 + 100 * 0.5e-6)

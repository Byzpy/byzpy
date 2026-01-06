import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from byzpy.utils.training import train_with_progress


@pytest.mark.asyncio
async def test_train_with_progress_basic_loop():
    """Test that the utility calls ps.round the correct number of times."""
    mock_ps = MagicMock()
    mock_ps.round = AsyncMock()

    rounds = 10
    history = await train_with_progress(mock_ps, rounds=rounds)

    # Verify ps.round() was called exactly 10 times
    assert mock_ps.round.call_count == rounds
    # Verify history is empty because no callback was provided
    assert history == []


@pytest.mark.asyncio
async def test_train_with_progress_with_callback():
    """Test that the eval_callback is called at the correct intervals."""
    mock_ps = MagicMock()
    mock_ps.round = AsyncMock()

    # Mock callback that returns dummy metrics
    async def mock_callback(round_num):
        return {"acc": 0.5 + (round_num / 100)}

    rounds = 100
    interval = 40

    history = await train_with_progress(
        mock_ps, rounds=rounds, eval_callback=mock_callback, eval_interval=interval
    )

    # At 100 rounds with interval 40, it should trigger at round 40 and 80
    assert len(history) == 2
    assert history[0]["round"] == 40
    assert history[1]["round"] == 80
    assert "acc" in history[0]["metrics"]


@pytest.mark.asyncio
async def test_train_with_progress_no_callback():
    """Test that it works fine when eval_callback is explicitly None."""
    mock_ps = MagicMock()
    mock_ps.round = AsyncMock()

    history = await train_with_progress(mock_ps, rounds=5, eval_callback=None)

    assert mock_ps.round.call_count == 5
    assert history == []

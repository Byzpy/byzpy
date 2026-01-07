import pytest

from byzpy.engine.parameter_server.ps import ParameterServer
from byzpy.utils.training import train_with_progress


class TestPS(ParameterServer):
    def __init__(self):
        pass

    async def round(self):
        """Simple override to make the round method do nothing."""
        return


@pytest.mark.asyncio
async def test_train_with_progress_basic_loop():
    """Test that the utility calls ps.round the correct number of times."""
    ps = TestPS()

    rounds = 5
    # history should be empty because no callback is provided
    history = await train_with_progress(ps, rounds=rounds)

    assert history == []


@pytest.mark.asyncio
async def test_train_with_progress_with_callback():
    """Test that the eval_callback is called at the correct intervals."""
    ps = TestPS()

    async def mock_callback(round_num):
        return {"acc": 0.5 + (round_num / 100)}

    rounds = 10
    interval = 5

    history = await train_with_progress(
        ps, rounds=rounds, eval_callback=mock_callback, eval_interval=interval
    )

    # At 10 rounds with interval 5, it should trigger at round 5 and 10
    assert len(history) == 2
    assert history[0]["round"] == 5
    assert history[1]["round"] == 10
    assert "acc" in history[0]["metrics"]


@pytest.mark.asyncio
async def test_train_with_progress_no_callback():
    """Test that it works fine when eval_callback is explicitly None."""
    ps = TestPS()

    history = await train_with_progress(ps, rounds=3, eval_callback=None)

    assert history == []

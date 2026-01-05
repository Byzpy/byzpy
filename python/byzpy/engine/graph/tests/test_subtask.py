from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from byzpy.engine.graph.subtask import SubTask


def test_subtask_defaults_are_immutable():
    subtask = SubTask(fn=lambda: None)

    assert subtask.args == ()
    assert subtask.kwargs == {}
    assert subtask.name is None
    assert subtask.affinity is None

    with pytest.raises(FrozenInstanceError):
        subtask.name = "new"  # type: ignore[misc]

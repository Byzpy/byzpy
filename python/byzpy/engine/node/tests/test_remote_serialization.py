"""Tests for Message Serialization - Category 4 from Milestone 5 Test Plan."""

import numpy as np
import pytest
import torch

from byzpy.engine.node.remote_client import deserialize_message, serialize_message

# =============================================================================
# Category 4.1: Basic Serialization
# =============================================================================


def test_serialize_simple_message():
    """Verify simple messages can be serialized for network transport."""
    msg = {"from": "node1", "type": "test", "payload": {"value": 42, "name": "test"}}

    serialized = serialize_message(msg)
    assert isinstance(serialized, bytes)

    deserialized = deserialize_message(serialized)
    assert deserialized == msg


def test_serialize_complex_message():
    """Verify complex nested messages can be serialized."""
    msg = {
        "from": "node1",
        "type": "complex",
        "payload": {"nested": {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}}},
    }

    serialized = serialize_message(msg)
    deserialized = deserialize_message(serialized)

    assert deserialized == msg


# =============================================================================
# Category 4.2: Tensor Serialization
# =============================================================================


def test_serialize_tensor_message():
    """Verify tensor messages can be serialized efficiently."""
    tensor = torch.randn(100, 100)
    msg = {"from": "node1", "type": "tensor", "payload": {"tensor": tensor}}

    serialized = serialize_message(msg)
    deserialized = deserialize_message(serialized)

    assert "tensor" in deserialized["payload"]
    torch.testing.assert_close(deserialized["payload"]["tensor"], tensor)


def test_serialize_numpy_array():
    """Verify numpy arrays can be serialized."""
    array = np.random.rand(50, 50).astype(np.float32)
    msg = {"from": "node1", "type": "array", "payload": {"array": array}}

    serialized = serialize_message(msg)
    deserialized = deserialize_message(serialized)

    assert "array" in deserialized["payload"]
    np.testing.assert_array_equal(deserialized["payload"]["array"], array)


def test_serialize_large_payload():
    """Verify large payloads can be serialized efficiently."""
    large_array = np.random.rand(1000, 1000).astype(np.float32)
    msg = {"from": "node1", "type": "large", "payload": {"data": large_array}}

    serialized = serialize_message(msg)
    deserialized = deserialize_message(serialized)

    np.testing.assert_array_equal(deserialized["payload"]["data"], large_array)

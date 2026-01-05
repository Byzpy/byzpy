from __future__ import annotations

import torch

from byzpy.engine.parameter_server.runner import ParameterServerRunner
from byzpy.engine.transport.tcp import TcpTransport


def test_parameter_server_runner_over_tcp():
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    transport = TcpTransport()
    runner = ParameterServerRunner(
        worker_grad_fns=[lambda g=g: g for g in grads], transport=transport
    )
    runner.start()
    try:
        out = runner.run_round()
        assert torch.allclose(out, sum(grads) / len(grads))
    finally:
        runner.stop()
        transport.close()
